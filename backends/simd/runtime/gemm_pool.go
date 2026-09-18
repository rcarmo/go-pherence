package simd

import (
	"context"
	"errors"
	"sync"

	"github.com/rcarmo/go-pherence/internal/checked"
)

var (
	ErrGEMMPoolClosed      = errors.New("simd: GEMM pool closed")
	ErrGEMMPoolNilContext  = errors.New("simd: nil context")
	ErrGEMMPoolInvalid     = errors.New("simd: invalid GEMM arguments")
	ErrGEMMPoolOversizedK  = errors.New("simd: GEMM k exceeds pool maxK")
	ErrGEMMPoolPackedSize  = errors.New("simd: packed NT weights buffer too short")
	ErrGEMMPoolPackedAlias = errors.New("simd: packed NT weights buffer must not overlap inputs")
	ErrGEMMPoolInternal    = errors.New("simd: GEMM pool worker rejected validated job")
)

// GEMMPool owns a fixed set of persistent workers for parallel NT GEMM.
//
// Run is serialized by the pool: at most one caller may execute matrix work at
// a time. Close is safe to call concurrently with Run; it prevents new runs,
// waits for the active one to finish, then shuts workers down. Close is
// idempotent.
//
// Cancellation is checked before each worker job is submitted and again after
// each submitted job completes. Submitted jobs are always waited for before Run
// returns. Individual SIMD kernels are not interruptible, so cancellation that
// arrives after a job starts may still leave the rows already assigned to that
// job mutated.
type GEMMPool struct {
	runMu   sync.Mutex
	stateMu sync.Mutex
	closed  bool
	closing bool

	maxK    int
	workers int

	directScratch []float32
	states        []gemmPoolWorker
	wg            sync.WaitGroup
}

type gemmPoolWorker struct {
	jobCh   chan gemmPoolJob
	doneCh  chan bool
	scratch []float32
}

type gemmPoolJob struct {
	c, a, w, packed []float32
	m, n, k         int
	alpha           float32
	lda, ldb, ldc   int
	prepacked       bool
}

type gemmPoolPartition int

const (
	gemmPoolPartitionRows gemmPoolPartition = iota
	gemmPoolPartitionColumns
)

// NewGEMMPool creates a persistent row-parallel NT GEMM pool.
func NewGEMMPool(workers, maxK int) (*GEMMPool, error) {
	if workers < 1 || workers > 64 || maxK < 1 {
		return nil, ErrGEMMPoolInvalid
	}
	scratchLen, ok := checked.MulInt(maxK, gebpNR)
	_, totalOK := checked.MulInt(scratchLen, workers*4)
	if !ok || !totalOK {
		return nil, ErrGEMMPoolOversizedK
	}
	p := &GEMMPool{
		maxK:    maxK,
		workers: workers,
	}
	if workers == 1 {
		p.directScratch = make([]float32, scratchLen)
		return p, nil
	}
	p.states = make([]gemmPoolWorker, workers)
	for i := range p.states {
		p.states[i] = gemmPoolWorker{
			jobCh:   make(chan gemmPoolJob, 1),
			doneCh:  make(chan bool, 1),
			scratch: make([]float32, scratchLen),
		}
		p.wg.Add(1)
		go p.worker(i)
	}
	return p, nil
}

// Close stops the pool. It is safe to call concurrently with Run and may be
// called more than once.
func (p *GEMMPool) Close() error {
	if p == nil {
		return nil
	}
	p.stateMu.Lock()
	if p.closed || p.closing {
		p.stateMu.Unlock()
		p.runMu.Lock()
		p.runMu.Unlock()
		p.wg.Wait()
		return nil
	}
	p.closing = true
	p.stateMu.Unlock()

	p.runMu.Lock()
	defer p.runMu.Unlock()

	p.stateMu.Lock()
	if p.closed {
		p.stateMu.Unlock()
		return nil
	}
	p.closed = true
	p.stateMu.Unlock()

	for i := range p.states {
		close(p.states[i].jobCh)
	}
	p.wg.Wait()
	return nil
}

// Run computes C += alpha*A*W^T using row-partitioned worker jobs. When packed
// is nil, workers use caller-owned streamed packing scratch that was
// preallocated at pool construction. When packed is non-nil, it must contain
// at least floor(n/16)*k*16 float32 values laid out as
// PackSgemmNTWeights/Into produce.
func (p *GEMMPool) Run(ctx context.Context, c, a, w, packed []float32, m, n, k int, alpha float32, lda, ldb, ldc int) error {
	return p.run(ctx, c, a, w, packed, m, n, k, alpha, lda, ldb, ldc, gemmPoolPartitionRows)
}

// RunColumns computes C += alpha*A*W^T using column-partitioned worker jobs.
// Only complete 16-column panels are split across workers; any final n%16 tail
// reuses the existing checked fallback path.
func (p *GEMMPool) RunColumns(ctx context.Context, c, a, w, packed []float32, m, n, k int, alpha float32, lda, ldb, ldc int) error {
	return p.run(ctx, c, a, w, packed, m, n, k, alpha, lda, ldb, ldc, gemmPoolPartitionColumns)
}

func (p *GEMMPool) run(ctx context.Context, c, a, w, packed []float32, m, n, k int, alpha float32, lda, ldb, ldc int, partition gemmPoolPartition) error {
	if p == nil {
		return ErrGEMMPoolClosed
	}
	if ctx == nil {
		return ErrGEMMPoolNilContext
	}
	p.runMu.Lock()
	defer p.runMu.Unlock()
	p.stateMu.Lock()
	closed := p.closed || p.closing
	p.stateMu.Unlock()
	if closed {
		return ErrGEMMPoolClosed
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	prepacked, err := p.validateRun(c, a, w, packed, m, n, k, lda, ldb, ldc)
	if err != nil {
		return err
	}
	if partition == gemmPoolPartitionColumns {
		return p.runColumnsChecked(ctx, c, a, w, packed, m, n, k, alpha, lda, ldb, ldc, prepacked)
	}
	return p.runRowsChecked(ctx, c, a, w, packed, m, n, k, alpha, lda, ldb, ldc, prepacked)
}

func (p *GEMMPool) runRowsChecked(ctx context.Context, c, a, w, packed []float32, m, n, k int, alpha float32, lda, ldb, ldc int, prepacked bool) error {
	fullRows := m / gebpMR * gebpMR
	fullTiles := fullRows / gebpMR
	useWorkers := p.workers
	if useWorkers > fullTiles {
		useWorkers = fullTiles
	}
	if useWorkers <= 1 {
		return p.runDirect(ctx, c, a, w, packed, m, n, k, alpha, lda, ldb, ldc, prepacked)
	}

	tileChunk := (fullTiles + useWorkers - 1) / useWorkers
	submitted := 0
	firstErr := error(nil)
	for worker := 0; worker < useWorkers; worker++ {
		if err := ctx.Err(); err != nil {
			firstErr = err
			break
		}
		row0 := worker * tileChunk * gebpMR
		if row0 >= fullRows {
			break
		}
		row1 := row0 + tileChunk*gebpMR
		if row1 > fullRows {
			row1 = fullRows
		}
		p.states[worker].jobCh <- gemmPoolJob{
			c:         c[row0*ldc:],
			a:         a[row0*lda:],
			w:         w,
			packed:    packed,
			m:         row1 - row0,
			n:         n,
			k:         k,
			alpha:     alpha,
			lda:       lda,
			ldb:       ldb,
			ldc:       ldc,
			prepacked: prepacked,
		}
		submitted++
	}
	for worker := 0; worker < submitted; worker++ {
		if ok := <-p.states[worker].doneCh; !ok && firstErr == nil {
			firstErr = ErrGEMMPoolInternal
		}
		if err := ctx.Err(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	if firstErr != nil {
		return firstErr
	}
	if fullRows < m {
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := p.runOne(c[fullRows*ldc:], a[fullRows*lda:], w, packed, m-fullRows, n, k, alpha, lda, ldb, ldc, prepacked, p.streamedScratch()); err != nil {
			return err
		}
		if err := ctx.Err(); err != nil {
			return err
		}
	}
	return nil
}

func (p *GEMMPool) runColumnsChecked(ctx context.Context, c, a, w, packed []float32, m, n, k int, alpha float32, lda, ldb, ldc int, prepacked bool) error {
	fullPanels := n / gebpNR
	useWorkers := p.workers
	if useWorkers > fullPanels {
		useWorkers = fullPanels
	}
	if useWorkers <= 1 {
		return p.runDirect(ctx, c, a, w, packed, m, n, k, alpha, lda, ldb, ldc, prepacked)
	}

	panelChunk := (fullPanels + useWorkers - 1) / useWorkers
	panelStride := k * gebpNR
	submitted := 0
	firstErr := error(nil)
	for worker := 0; worker < useWorkers; worker++ {
		if err := ctx.Err(); err != nil {
			firstErr = err
			break
		}
		panel0 := worker * panelChunk
		if panel0 >= fullPanels {
			break
		}
		panel1 := panel0 + panelChunk
		if panel1 > fullPanels {
			panel1 = fullPanels
		}
		col0 := panel0 * gebpNR
		cols := (panel1 - panel0) * gebpNR
		jobPacked := packed
		if prepacked {
			// Each full panel contributes k*16 entries, so col0*k == panel0*panelStride.
			jobPacked = packed[panel0*panelStride:]
		}
		p.states[worker].jobCh <- gemmPoolJob{
			c:         c[col0:],
			a:         a,
			w:         w[col0*ldb:],
			packed:    jobPacked,
			m:         m,
			n:         cols,
			k:         k,
			alpha:     alpha,
			lda:       lda,
			ldb:       ldb,
			ldc:       ldc,
			prepacked: prepacked,
		}
		submitted++
	}
	for worker := 0; worker < submitted; worker++ {
		if ok := <-p.states[worker].doneCh; !ok && firstErr == nil {
			firstErr = ErrGEMMPoolInternal
		}
		if err := ctx.Err(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	if firstErr != nil {
		return firstErr
	}
	fullCols := fullPanels * gebpNR
	if fullCols < n {
		if err := ctx.Err(); err != nil {
			return err
		}
		tailPacked := packed
		if prepacked {
			tailPacked = packed[fullPanels*panelStride:]
		}
		if err := p.runOne(c[fullCols:], a, w[fullCols*ldb:], tailPacked, m, n-fullCols, k, alpha, lda, ldb, ldc, prepacked, p.streamedScratch()); err != nil {
			return err
		}
		if err := ctx.Err(); err != nil {
			return err
		}
	}
	return nil
}

func (p *GEMMPool) worker(i int) {
	defer p.wg.Done()
	w := &p.states[i]
	for job := range w.jobCh {
		w.doneCh <- p.runOneBool(job, w.scratch)
	}
}

func (p *GEMMPool) validateRun(c, a, w, packed []float32, m, n, k, lda, ldb, ldc int) (bool, error) {
	if !validSgemmSliceArgs(c, a, w, m, n, k, lda, ldb, ldc, true) {
		return false, ErrGEMMPoolInvalid
	}
	if !float32SlicesDisjoint(c, a) || !float32SlicesDisjoint(c, w) {
		return false, ErrGEMMPoolInvalid
	}
	if k > p.maxK {
		return false, ErrGEMMPoolOversizedK
	}
	if packed == nil {
		return false, nil
	}
	_, packedLen, ok := checkedSgemmNTFullPanelLayout(n, k)
	if !ok {
		return false, ErrGEMMPoolInvalid
	}
	if len(packed) < packedLen {
		return false, ErrGEMMPoolPackedSize
	}
	if packedLen > 0 {
		pp := packed[:packedLen]
		if !float32SlicesDisjoint(pp, c) || !float32SlicesDisjoint(pp, a) || !float32SlicesDisjoint(pp, w) {
			return false, ErrGEMMPoolPackedAlias
		}
	}
	return true, nil
}

func (p *GEMMPool) runDirect(ctx context.Context, c, a, w, packed []float32, m, n, k int, alpha float32, lda, ldb, ldc int, prepacked bool) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := p.runOne(c, a, w, packed, m, n, k, alpha, lda, ldb, ldc, prepacked, p.streamedScratch()); err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	return nil
}

func (p *GEMMPool) runOneBool(job gemmPoolJob, scratch []float32) bool {
	return p.runOne(job.c, job.a, job.w, job.packed, job.m, job.n, job.k, job.alpha, job.lda, job.ldb, job.ldc, job.prepacked, scratch) == nil
}

func (p *GEMMPool) runOne(c, a, w, packed []float32, m, n, k int, alpha float32, lda, ldb, ldc int, prepacked bool, scratch []float32) error {
	var ok bool
	if prepacked {
		ok = SgemmNTPrepackedTo(c, a, w, packed, m, n, k, alpha, lda, ldb, ldc)
	} else {
		ok = SgemmNTPackedTo(c, a, w, scratch, m, n, k, alpha, lda, ldb, ldc)
	}
	if !ok {
		return ErrGEMMPoolInternal
	}
	return nil
}

func (p *GEMMPool) streamedScratch() []float32 {
	if p.directScratch != nil {
		return p.directScratch
	}
	return p.states[0].scratch
}
