package main

import (
	"fmt"
	"os"
	"runtime"
	"strconv"
	"sync"
	"syscall"
	"unsafe"

	"github.com/rcarmo/go-pherence/backends/spacemit/ime2"
	tcmpkg "github.com/rcarmo/go-pherence/backends/spacemit/tcm"
	"golang.org/x/sys/unix"
)

// AIWorkerPool manages persistent goroutines on AI cores (8-15).
// Workers are created ONCE and stay alive for the lifetime of inference.
type AIWorkerPool struct {
	n         int
	tasks     []chan aiTask
	done      []chan struct{}
	tcm       *tcmpkg.TCM
	tcmSlices [][]byte
}

type aiTask struct {
	fn func()
}

// NewAIWorkerPool creates n persistent AI workers pinned to cores 8-15.
func NewAIWorkerPool(n int) *AIWorkerPool {
	p := &AIWorkerPool{
		n:     n,
		tasks: make([]chan aiTask, n),
		done:  make([]chan struct{}, n),
	}
	if os.Getenv("IME2_TCM_ACT") != "" && tcmpkg.IsAvailable() {
		if dev, err := tcmpkg.Open(); err == nil {
			p.tcm = dev
			p.tcmSlices = make([][]byte, n)
			for i := 0; i < n; i++ {
				p.tcmSlices[i] = dev.Slice(i % tcmpkg.BlockCount)
			}
			fmt.Fprintf(os.Stderr, "AI worker pool: TCM activation staging enabled (%d blocks)\n", n)
		} else {
			fmt.Fprintf(os.Stderr, "AI worker pool: TCM activation staging unavailable: %v\n", err)
		}
	}
	var wg sync.WaitGroup
	wg.Add(n)
	for i := 0; i < n; i++ {
		p.tasks[i] = make(chan aiTask, 1)
		p.done[i] = make(chan struct{}, 1)
		go func(id int) {
			// Pin to AI core PERMANENTLY
			runtime.LockOSThread()
			tid := syscall.Gettid()
			if f, err := os.OpenFile("/proc/set_ai_thread", os.O_WRONLY, 0); err == nil {
				f.Write([]byte(strconv.Itoa(tid)))
				f.Close()
			}
			var set unix.CPUSet
			set.Zero()
			set.Set(8 + id%8)
			unix.SchedSetaffinity(0, &set)
			wg.Done()
			// Process tasks forever
			for task := range p.tasks[id] {
				task.fn()
				p.done[id] <- struct{}{}
			}
		}(i)
	}
	wg.Wait() // ensure all workers are registered before returning
	fmt.Fprintf(os.Stderr, "AI worker pool: %d workers on cores 8-%d\n", n, 7+n)
	return p
}

// Run dispatches work to all workers and waits for completion.
func (p *AIWorkerPool) Run(fn func(workerID, nWorkers int)) {
	for i := 0; i < p.n; i++ {
		i := i
		p.tasks[i] <- aiTask{fn: func() { fn(i, p.n) }}
	}
	for i := 0; i < p.n; i++ {
		<-p.done[i]
	}
}

// Close shuts down all workers.
func (p *AIWorkerPool) Close() {
	for i := 0; i < p.n; i++ {
		close(p.tasks[i])
	}
	if p.tcm != nil {
		p.tcm.Close()
		p.tcm = nil
	}
}

// aiGemmSpec describes one M×K matmul using already-packed weight and
// activation layouts.
type aiGemmSpec struct {
	M, K      int
	wPacked   []int8
	actPacked []int8
	wScale    float32
	actScale  float32
	out       []float32
}

// runAIGemmWorker executes one GEMM shard for a single AI worker.
func runAIGemmWorker(spec aiGemmSpec, workerID, nWorkers int) {
	runAIGemmWorkerWithAct(spec, workerID, nWorkers, spec.actPacked)
}

func runAIGemmWorkerWithAct(spec aiGemmSpec, workerID, nWorkers int, actPacked []int8) {
	tilesPerRow := spec.K / 16
	combined := spec.wScale * spec.actScale
	rowStart := (workerID * spec.M / nWorkers / 8) * 8
	rowEnd := ((workerID + 1) * spec.M / nWorkers / 8) * 8
	if workerID == nWorkers-1 {
		rowEnd = spec.M
	}
	for i := rowStart; i < rowEnd; i += 8 {
		var acc [64]int32
		ime2.VmadotKLoop1024(
			(*byte)(unsafe.Pointer(&spec.wPacked[(i/8)*tilesPerRow*128])),
			(*byte)(unsafe.Pointer(&actPacked[0])),
			&acc[0], spec.K,
		)
		for r := 0; r < 8 && i+r < spec.M; r++ {
			spec.out[i+r] = float32(acc[r*8]) * combined
		}
	}
}

// GemmAIPooled performs M×K matmul distributed across the AI worker pool.
func GemmAIPooled(M, K int, wPacked, actPacked []int8, wScale, actScale float32, out []float32, pool *AIWorkerPool) {
	GemmAIPooledBatch(pool, aiGemmSpec{M: M, K: K, wPacked: wPacked, actPacked: actPacked, wScale: wScale, actScale: actScale, out: out})
}

// runAIGemmWorkerVL32 executes one GEMM shard with the known-good forced-vl=32
// AI-core kernel and legacy 4×8 tile layout.
func runAIGemmWorkerVL32(spec aiGemmSpec, workerID, nWorkers int) {
	tilesPerRow := spec.K / 8
	combined := spec.wScale * spec.actScale
	rowStart := (workerID * spec.M / nWorkers / 4) * 4
	rowEnd := ((workerID + 1) * spec.M / nWorkers / 4) * 4
	if workerID == nWorkers-1 {
		rowEnd = spec.M
	}
	for i := rowStart; i < rowEnd; i += 4 {
		var acc [16]int32
		ime2.VmadotKLoopAI(
			(*byte)(unsafe.Pointer(&spec.wPacked[(i/4)*tilesPerRow*32])),
			(*byte)(unsafe.Pointer(&spec.actPacked[0])),
			&acc[0], spec.K,
		)
		for r := 0; r < 4 && i+r < spec.M; r++ {
			spec.out[i+r] = float32(acc[r*4]) * combined
		}
	}
}

// GemmAIPooledVL32 performs M×K matmul on A100 workers with forced vl=32.
func GemmAIPooledVL32(M, K int, wPacked, actPacked []int8, wScale, actScale float32, out []float32, pool *AIWorkerPool) {
	GemmAIPooledBatchVL32(pool, aiGemmSpec{M: M, K: K, wPacked: wPacked, actPacked: actPacked, wScale: wScale, actScale: actScale, out: out})
}

// GemmAIPooledBatch performs multiple independent matmuls in one worker-pool
// dispatch. This reduces channel/barrier overhead for Q/K/V and Gate/Up pairs,
// which otherwise launch hundreds of tiny AI-core jobs per decoded token.
func GemmAIPooledBatch(pool *AIWorkerPool, specs ...aiGemmSpec) {
	pool.Run(func(workerID, nWorkers int) {
		for _, spec := range specs {
			actPacked := spec.actPacked
			if pool.tcmSlices != nil && workerID < len(pool.tcmSlices) {
				need := len(spec.actPacked)
				if need > 0 && need <= len(pool.tcmSlices[workerID]) {
					buf := pool.tcmSlices[workerID][:need]
					copy(buf, *(*[]byte)(unsafe.Pointer(&spec.actPacked)))
					actPacked = *(*[]int8)(unsafe.Pointer(&buf))
				}
			}
			runAIGemmWorkerWithAct(spec, workerID, nWorkers, actPacked)
		}
	})
}

// GemmAIPooledBatchVL32 is the forced-vl=32 equivalent of GemmAIPooledBatch.
func GemmAIPooledBatchVL32(pool *AIWorkerPool, specs ...aiGemmSpec) {
	pool.Run(func(workerID, nWorkers int) {
		for _, spec := range specs {
			runAIGemmWorkerVL32(spec, workerID, nWorkers)
		}
	})
}
