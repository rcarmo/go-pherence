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
	"golang.org/x/sys/unix"
)

// AIWorkerPool manages persistent goroutines on AI cores (8-15).
// Workers are created ONCE and stay alive for the lifetime of inference.
type AIWorkerPool struct {
	n     int
	tasks []chan aiTask
	done  []chan struct{}
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
}

// GemmAIPooled performs M×K matmul distributed across the AI worker pool.
func GemmAIPooled(M, K int, wPacked, actPacked []int8, wScale, actScale float32, out []float32, pool *AIWorkerPool) {
	tilesPerRow := K / 32
	combined := wScale * actScale
	pool.Run(func(workerID, nWorkers int) {
		rowStart := (workerID * M / nWorkers / 4) * 4
		rowEnd := ((workerID + 1) * M / nWorkers / 4) * 4
		if workerID == nWorkers-1 { rowEnd = M }
		for i := rowStart; i < rowEnd; i += 4 {
			var acc [64]int32
			ime2.VmadotKLoop1024(
				(*byte)(unsafe.Pointer(&wPacked[(i/4)*tilesPerRow*128])),
				(*byte)(unsafe.Pointer(&actPacked[0])),
				&acc[0], K,
			)
			for r := 0; r < 4 && i+r < M; r++ {
				out[i+r] = float32(acc[r*16]+acc[r*16+1]) * combined
			}
		}
	})
}
