package ime2

import (
	"runtime"
	"sync"
	"unsafe"

	"golang.org/x/sys/unix"
)

// WorkerPool maintains persistent goroutines pinned to X100 cores.
// Eliminates goroutine spawn overhead for per-matmul parallelism.
type WorkerPool struct {
	n       int
	tasks   []chan workItem
	done    []chan struct{}
}

type workItem struct {
	fn func()
}

// NewWorkerPool creates n worker goroutines, each pinned to a core.
func NewWorkerPool(n int) *WorkerPool {
	p := &WorkerPool{
		n:     n,
		tasks: make([]chan workItem, n),
		done:  make([]chan struct{}, n),
	}
	for i := 0; i < n; i++ {
		p.tasks[i] = make(chan workItem, 1)
		p.done[i] = make(chan struct{}, 1)
		go p.worker(i)
	}
	return p
}

func (p *WorkerPool) worker(id int) {
	runtime.LockOSThread()
	var cpuSet unix.CPUSet
	cpuSet.Zero()
	cpuSet.Set(id % 8)
	unix.SchedSetaffinity(0, &cpuSet)

	for task := range p.tasks[id] {
		task.fn()
		p.done[id] <- struct{}{}
	}
}

// Run dispatches fn(workerID) to all workers and waits for completion.
func (p *WorkerPool) Run(fn func(workerID, nWorkers int)) {
	for i := 0; i < p.n; i++ {
		i := i
		p.tasks[i] <- workItem{fn: func() { fn(i, p.n) }}
	}
	for i := 0; i < p.n; i++ {
		<-p.done[i]
	}
}

// Close shuts down all workers.
func (p *WorkerPool) Close() {
	for i := 0; i < p.n; i++ {
		close(p.tasks[i])
	}
}

// GemmINT8PackedPool performs C[M×N] = A * B^T using a persistent worker pool.
func GemmINT8PackedPool(M, N, K int, Apacked, Bpacked []int8, C []int32, pool *WorkerPool) {
	tilesPerRow := K / 8

	pool.Run(func(workerID, nWorkers int) {
		rowsPerWorker := ((M / 4) / nWorkers) * 4
		iStart := workerID * rowsPerWorker
		iEnd := iStart + rowsPerWorker
		if workerID == nWorkers-1 {
			iEnd = M
		}
		if iStart >= M {
			return
		}

		for i := iStart; i < iEnd; i += 4 {
			aBase := (i / 4) * tilesPerRow * 32
			for j := 0; j < N; j += 4 {
				bBase := (j / 4) * tilesPerRow * 32
				var acc [16]int32
				vmadotKLoop(
					(*byte)(unsafe.Pointer(&Apacked[aBase])),
					(*byte)(unsafe.Pointer(&Bpacked[bBase])),
					&acc[0],
					K,
				)
				for r := 0; r < 4; r++ {
					for c := 0; c < 4; c++ {
						C[(i+r)*N+(j+c)] = acc[r*4+c]
					}
				}
			}
		}
	})
}

var _ sync.Mutex // ensure import



// CondPool uses sync.Cond for lower-latency dispatch than channels.
type CondPool struct {
	n      int
	mu     sync.Mutex
	cond   *sync.Cond
	fn     func(int, int)
	phase  int
	done   int
}

func NewCondPool(n int) *CondPool {
	p := &CondPool{n: n}
	p.cond = sync.NewCond(&p.mu)
	for i := 0; i < n; i++ {
		go p.condWorker(i)
	}
	return p
}

func (p *CondPool) condWorker(id int) {
	runtime.LockOSThread()
	var cpuSet unix.CPUSet
	cpuSet.Zero()
	cpuSet.Set(id % 8)
	unix.SchedSetaffinity(0, &cpuSet)
	myPhase := 0
	for {
		p.mu.Lock()
		for p.phase == myPhase { p.cond.Wait() }
		myPhase = p.phase
		fn := p.fn
		p.mu.Unlock()
		if fn == nil { return }
		fn(id, p.n)
		p.mu.Lock()
		p.done++
		if p.done == p.n { p.cond.Broadcast() }
		p.mu.Unlock()
	}
}

func (p *CondPool) Run(fn func(workerID, nWorkers int)) {
	p.mu.Lock()
	p.fn = fn
	p.done = 0
	p.phase++
	p.cond.Broadcast()
	for p.done < p.n { p.cond.Wait() }
	p.mu.Unlock()
}

func (p *CondPool) Close() {
	p.mu.Lock()
	p.fn = nil
	p.phase++
	p.cond.Broadcast()
	p.mu.Unlock()
}
