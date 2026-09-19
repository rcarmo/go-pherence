//go:build linux && riscv64

package aipool

import (
	"fmt"
	"os"
	"runtime"
	"strconv"
	"sync"
	"sync/atomic"
	"syscall"
	"unsafe"

	tcmpkg "github.com/rcarmo/go-pherence/backends/spacemit/tcm"
	"golang.org/x/sys/unix"
)

// AIWorkerPool manages persistent goroutines on AI cores (8-15).
// Uses atomic spin with periodic yields for dispatch signaling.
type AIWorkerPool struct {
	runMu   sync.Mutex // Run/Close ownership; callbacks must not re-enter
	workers sync.WaitGroup
	N       int
	// Dispatch: caller sets fn + increments gen; workers spin on gen
	gen  atomic.Int64
	fn   unsafe.Pointer // *func(int,int)
	done atomic.Int64
	stop atomic.Int64

	tcm       *tcmpkg.TCM
	TcmSlices [][]byte
}

func NewAIWorkerPool(n int) *AIWorkerPool {
	// There are eight AI cores/TCM blocks; sharing a block concurrently aliases
	// activation scratch. Existing callers already cap workers to this range.
	if n < 1 {
		n = 1
	}
	if n > tcmpkg.BlockCount {
		n = tcmpkg.BlockCount
	}
	p := &AIWorkerPool{N: n}
	if os.Getenv("IME2_TCM_ACT") != "0" && tcmpkg.IsAvailable() {
		if dev, err := tcmpkg.Open(); err == nil {
			p.tcm = dev
			p.TcmSlices = make([][]byte, n)
			for i := 0; i < n; i++ {
				p.TcmSlices[i] = dev.Slice(i % tcmpkg.BlockCount)
			}
			fmt.Fprintf(os.Stderr, "AI worker pool: TCM activation staging enabled (%d blocks)\n", n)
		}
	}
	var ready atomic.Int64
	for i := 0; i < n; i++ {
		p.workers.Add(1)
		go func(id int) {
			defer p.workers.Done()
			runtime.LockOSThread()
			tid := syscall.Gettid()
			if f, err := os.OpenFile("/proc/set_ai_thread", os.O_WRONLY, 0); err == nil {
				_, _ = f.Write([]byte(strconv.Itoa(tid)))
				f.Close()
			}
			var set unix.CPUSet
			set.Zero()
			set.Set(8 + id%8)
			unix.SchedSetaffinity(0, &set)
			ready.Add(1)

			lastGen := int64(0)
			for {
				// Wait for new work
				for {
					if p.stop.Load() != 0 {
						return
					}
					g := p.gen.Load()
					if g != lastGen {
						lastGen = g
						break
					}
					runtime.Gosched()
				}
				// Execute
				fnPtr := (*func(int, int))(atomic.LoadPointer(&p.fn))
				if fnPtr != nil {
					(*fnPtr)(id, n)
				}
				p.done.Add(1)
			}
		}(i)
	}
	for ready.Load() < int64(n) {
		runtime.Gosched()
	}
	fmt.Fprintf(os.Stderr, "AI worker pool: %d workers on cores 8-%d\n", n, 7+n)
	return p
}

// Run dispatches fn to all workers and waits for completion.
func (p *AIWorkerPool) Run(fn func(workerID, nWorkers int)) {
	if p == nil || fn == nil {
		return
	}
	p.runMu.Lock()
	defer p.runMu.Unlock()
	if p.stop.Load() != 0 {
		return
	}
	p.done.Store(0)
	atomic.StorePointer(&p.fn, unsafe.Pointer(&fn))
	p.gen.Add(1)
	target := int64(p.N)
	for p.done.Load() < target {
		runtime.Gosched()
	}
	atomic.StorePointer(&p.fn, nil)
}

// Close drains admitted work, joins affinity-modified workers, then unmaps TCM.
// It is idempotent; Run after Close is a no-op. Do not call from a worker callback.
func (p *AIWorkerPool) Close() {
	if p == nil {
		return
	}
	p.runMu.Lock()
	defer p.runMu.Unlock()
	if p.stop.Load() != 0 {
		return
	}
	p.stop.Store(1)
	p.workers.Wait()
	if p.tcm != nil {
		p.tcm.Close()
		p.tcm = nil
	}
	p.TcmSlices = nil
}
