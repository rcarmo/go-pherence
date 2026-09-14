package omnivoice

import (
	"fmt"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

func (b *Backbone) workerMaxK() int {
	if b == nil || b.weights == nil {
		return 0
	}
	cfg := b.weights.Config.LLMConfig
	if cfg.HiddenSize > cfg.IntermediateSize {
		return cfg.HiddenSize
	}
	return cfg.IntermediateSize
}

// EnableWorkers installs a persistent GEMM pool for block projections and the
// audio-head projection. The pool is sized once for max(hidden, intermediate),
// and workers == 1 still enables the pool's direct single-worker path.
//
// Worker pools are borrowed by siblings created after this call and are not
// reference counted. Repeating the same configuration is a no-op; changing the
// worker count after attachment is rejected so borrowed siblings never observe a
// silently replaced pool. Construct a new backbone if a different size is
// needed. Like ForwardInto, this method requires exclusive access to the
// backbone; do not call it concurrently with inference or Close.
func (b *Backbone) EnableWorkers(workers int) error {
	if b == nil || b.weights == nil {
		return fmt.Errorf("omnivoice: nil backbone")
	}
	if workers < 1 {
		return fmt.Errorf("omnivoice: workers must be >= 1")
	}
	if b.pool != nil {
		if b.poolWorkers == workers {
			return nil
		}
		return fmt.Errorf("omnivoice: worker pool already configured for %d workers", b.poolWorkers)
	}
	pool, err := simd.NewGEMMPool(workers, b.workerMaxK())
	if err != nil {
		return err
	}
	b.pool = pool
	b.poolWorkers = workers
	b.ownsPool = true
	return nil
}

// Close detaches any worker pool from this backbone and closes it only when the
// pool is owned here. Borrowed sibling pools are simply forgotten; the owner is
// responsible for closing them after all borrowers are done. All inference and
// configuration calls on this backbone must finish before Close. Unlike the
// underlying GEMMPool, Backbone is not safe for concurrent lifecycle calls.
func (b *Backbone) Close() error {
	if b == nil {
		return nil
	}
	pool := b.pool
	owns := b.ownsPool
	b.pool = nil
	b.poolWorkers = 0
	b.ownsPool = false
	if !owns || pool == nil {
		return nil
	}
	return pool.Close()
}

// EnableColumnWorkers enables opt-in panel-parallel GEMM on this backbone.
// Call before creating siblings; existing siblings retain their own selection.
func (b *Backbone) EnableColumnWorkers(workers int) error {
	if err := b.EnableWorkers(workers); err != nil {
		return err
	}
	b.columnWorkers = true
	return nil
}
