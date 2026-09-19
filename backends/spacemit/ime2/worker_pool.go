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
	runMu   sync.Mutex // serialises Run and Close; callbacks must not re-enter
	workers sync.WaitGroup
	closed  bool
	n       int
	tasks   []chan workItem
	done    []chan struct{}
}

type workItem struct {
	fn func()
}

// NewWorkerPool creates n worker goroutines, each pinned to a core.
func NewWorkerPool(n int) *WorkerPool { return newWorkerPool(n, pinWorker) }

// setup is nil only in lifecycle tests; tests exercise dispatch, not IME kernels.
func newWorkerPool(n int, setup func(int)) *WorkerPool {
	if n < 0 {
		n = 0
	}
	p := &WorkerPool{
		n:     n,
		tasks: make([]chan workItem, n),
		done:  make([]chan struct{}, n),
	}
	for i := 0; i < n; i++ {
		p.tasks[i] = make(chan workItem, 1)
		p.done[i] = make(chan struct{}, 1)
		p.workers.Add(1)
		go p.worker(i, setup)
	}
	return p
}

// The affinity-modified OS thread is discarded when its goroutine exits; do
// not unlock it and return a pinned-to-one-core thread to Go's shared pool.
func pinWorker(id int) {
	runtime.LockOSThread()
	var cpuSet unix.CPUSet
	cpuSet.Zero()
	cpuSet.Set(id % 8)
	_ = unix.SchedSetaffinity(0, &cpuSet)
}

func (p *WorkerPool) worker(id int, setup func(int)) {
	defer p.workers.Done()
	if setup != nil {
		setup(id)
	}
	for task := range p.tasks[id] {
		task.fn()
		p.done[id] <- struct{}{}
	}
}

// Run dispatches fn(workerID) to all workers and waits for completion.
func (p *WorkerPool) Run(fn func(workerID, nWorkers int)) {
	if p == nil || fn == nil {
		return
	}
	p.runMu.Lock()
	defer p.runMu.Unlock()
	if p.closed {
		return
	}
	for i := 0; i < p.n; i++ {
		i := i
		p.tasks[i] <- workItem{fn: func() { fn(i, p.n) }}
	}
	for i := 0; i < p.n; i++ {
		<-p.done[i]
	}
}

// Close waits for admitted work and joins every worker. It is idempotent.
// Run after Close is a no-op. Neither method may be called from a callback.
func (p *WorkerPool) Close() {
	if p == nil {
		return
	}
	p.runMu.Lock()
	defer p.runMu.Unlock()
	if p.closed {
		return
	}
	p.closed = true
	for i := 0; i < p.n; i++ {
		close(p.tasks[i])
	}
	p.workers.Wait()
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

// CondPool uses sync.Cond for lower-latency dispatch than channels.
type CondPool struct {
	runMu   sync.Mutex
	workers sync.WaitGroup
	closed  bool
	n       int
	mu      sync.Mutex
	cond    *sync.Cond
	fn      func(int, int)
	phase   int
	done    int
}

func NewCondPool(n int) *CondPool { return newCondPool(n, pinWorker) }
func newCondPool(n int, setup func(int)) *CondPool {
	if n < 0 {
		n = 0
	}
	p := &CondPool{n: n}
	p.cond = sync.NewCond(&p.mu)
	for i := 0; i < n; i++ {
		p.workers.Add(1)
		go p.condWorker(i, setup)
	}
	return p
}

func (p *CondPool) condWorker(id int, setup func(int)) {
	defer p.workers.Done()
	if setup != nil {
		setup(id)
	}
	myPhase := 0
	for {
		p.mu.Lock()
		for p.phase == myPhase {
			p.cond.Wait()
		}
		myPhase = p.phase
		fn := p.fn
		p.mu.Unlock()
		if fn == nil {
			return
		}
		fn(id, p.n)
		p.mu.Lock()
		p.done++
		if p.done == p.n {
			p.cond.Broadcast()
		}
		p.mu.Unlock()
	}
}

func (p *CondPool) Run(fn func(workerID, nWorkers int)) {
	if p == nil || fn == nil {
		return
	}
	p.runMu.Lock()
	defer p.runMu.Unlock()
	if p.closed {
		return
	}
	p.mu.Lock()
	p.fn = fn
	p.done = 0
	p.phase++
	p.cond.Broadcast()
	for p.done < p.n {
		p.cond.Wait()
	}
	p.fn = nil // do not retain caller captures until the next dispatch
	p.mu.Unlock()
}

// Close drains admitted Run calls and joins workers; repeated Close/Run-after-
// Close are safe. Callbacks must not re-enter either lifecycle method.
func (p *CondPool) Close() {
	if p == nil {
		return
	}
	p.runMu.Lock()
	defer p.runMu.Unlock()
	if p.closed {
		return
	}
	p.closed = true
	p.mu.Lock()
	p.fn = nil
	p.phase++
	p.cond.Broadcast()
	p.mu.Unlock()
	p.workers.Wait()
}
