// Package resourcebudget provides cooperative, process-local CPU-slot and
// estimated-memory reservations. It neither starts work nor enforces OS limits.
package resourcebudget

import (
	"context"
	"errors"
	"sync"
)

var (
	ErrInvalid  = errors.New("resourcebudget: invalid configuration or reservation")
	ErrCapacity = errors.New("resourcebudget: request exceeds total capacity")
	ErrBusy     = errors.New("resourcebudget: waiting limit reached")
	ErrClosed   = errors.New("resourcebudget: admission closed")
)

// Resources describes declared demand, not measured CPU utilisation or RSS.
// One dimension may be zero (e.g. a resident model holds memory without CPU).
type Resources struct {
	CPUSlots    int
	MemoryBytes int64
}
type Config struct {
	Capacity              Resources
	MaxActive, MaxWaiting int
}

// Snapshot contains only scalar detached counters. Closed admission can still
// have active leases, which must drain before their owners tear resources down.
type Snapshot struct {
	Capacity, Used  Resources
	Active, Waiting int
	Closed          bool
}

type Budget struct {
	mu      sync.Mutex
	cfg     Config
	used    Resources
	active  int
	waiting []*waiter
	closed  bool
	drained chan struct{}
}
type waiter struct {
	ctx   context.Context
	need  Resources
	ready chan struct{}
	lease *Lease
	err   error
}

// Lease must not be copied. Context expiry never revokes a granted lease: its
// owner releases AFTER all callbacks/native/device work and resource use drain.
// Release is idempotent; concurrent Shrink/Release are serialised by the budget.
// noCopy makes accidental value copies visible to go vet (as for sync.Mutex).
type noCopy struct{}

func (*noCopy) Lock()   {}
func (*noCopy) Unlock() {}

type Lease struct {
	_        noCopy
	budget   *Budget
	need     Resources
	released bool
}

func New(c Config) (*Budget, error) {
	if c.Capacity.CPUSlots < 1 || c.Capacity.MemoryBytes < 1 || c.MaxActive < 1 || c.MaxActive > 65536 || c.MaxWaiting < 0 || c.MaxWaiting > 65536 {
		return nil, ErrInvalid
	}
	return &Budget{cfg: c, waiting: make([]*waiter, 0, c.MaxWaiting), drained: make(chan struct{})}, nil
}
func valid(r Resources) bool {
	return r.CPUSlots >= 0 && r.MemoryBytes >= 0 && (r.CPUSlots > 0 || r.MemoryBytes > 0)
}
func (b *Budget) fits(r Resources) bool {
	return b.active < b.cfg.MaxActive && r.CPUSlots <= b.cfg.Capacity.CPUSlots-b.used.CPUSlots && r.MemoryBytes <= b.cfg.Capacity.MemoryBytes-b.used.MemoryBytes
}
func (b *Budget) grant(r Resources) *Lease {
	b.used.CPUSlots += r.CPUSlots
	b.used.MemoryBytes += r.MemoryBytes
	b.active++
	return &Lease{budget: b, need: r}
}

// Acquire atomically reserves BOTH dimensions or neither. FIFO is defined by
// registration under the mutex; small requests never bypass a waiting head.
// Cancelled waiters are removed. Fast uncontended acquisition creates no channel
// or goroutine. A caller must not wait while holding a lease it needs to release
// for the new request to fit: nested acquisition can deadlock at the owner level.
func (b *Budget) Acquire(ctx context.Context, r Resources) (*Lease, error) {
	if ctx == nil || !valid(r) {
		return nil, ErrInvalid
	}
	if e := ctx.Err(); e != nil {
		return nil, e
	}
	b.mu.Lock()
	if e := ctx.Err(); e != nil {
		b.mu.Unlock()
		return nil, e
	}
	if b.closed {
		b.mu.Unlock()
		return nil, ErrClosed
	}
	if r.CPUSlots > b.cfg.Capacity.CPUSlots || r.MemoryBytes > b.cfg.Capacity.MemoryBytes {
		b.mu.Unlock()
		return nil, ErrCapacity
	}
	// Reap cancelled waiters before deciding FIFO or pending capacity.
	b.dispatch()
	if len(b.waiting) == 0 && b.fits(r) {
		lease := b.grant(r)
		b.mu.Unlock()
		return lease, nil
	}
	if len(b.waiting) >= b.cfg.MaxWaiting {
		b.mu.Unlock()
		return nil, ErrBusy
	}
	w := &waiter{ctx: ctx, need: r, ready: make(chan struct{})}
	b.waiting = append(b.waiting, w)
	b.mu.Unlock()
	select {
	case <-w.ready:
	case <-ctx.Done():
	}
	b.mu.Lock()
	if w.lease != nil {
		// A grant can race cancellation. Return that ownership only if context is
		// still live; otherwise release it here before returning the cancellation.
		if e := ctx.Err(); e != nil {
			b.release(w.lease)
			b.mu.Unlock()
			return nil, e
		}
		lease := w.lease
		b.mu.Unlock()
		return lease, nil
	}
	if w.err != nil {
		e := w.err
		b.mu.Unlock()
		return nil, e
	}
	for i, x := range b.waiting {
		if x == w {
			b.remove(i)
			break
		}
	}
	e := ctx.Err()
	b.dispatch()
	b.mu.Unlock()
	return nil, e
}
func (b *Budget) remove(i int) {
	copy(b.waiting[i:], b.waiting[i+1:])
	last := len(b.waiting) - 1
	b.waiting[last] = nil
	b.waiting = b.waiting[:last]
}

// dispatch runs only while locked and never invokes caller callbacks.
func (b *Budget) dispatch() {
	if b.closed {
		return
	}
	// Compact cancelled entries once; repeated remove/copy would be quadratic
	// when many waiters cancel together. Clear tail pointers to drop contexts.
	kept := 0
	for _, w := range b.waiting {
		if e := w.ctx.Err(); e != nil {
			w.err = e
			close(w.ready)
		} else {
			b.waiting[kept] = w
			kept++
		}
	}
	clear(b.waiting[kept:])
	b.waiting = b.waiting[:kept]
	granted := 0
	for _, w := range b.waiting {
		if !b.fits(w.need) {
			break
		}
		w.lease = b.grant(w.need)
		close(w.ready)
		granted++
	}
	if granted > 0 {
		remaining := copy(b.waiting, b.waiting[granted:])
		clear(b.waiting[remaining:])
		b.waiting = b.waiting[:remaining]
	}
}
func (b *Budget) release(l *Lease) {
	if l.released {
		return
	}
	l.released = true
	b.used.CPUSlots -= l.need.CPUSlots
	b.used.MemoryBytes -= l.need.MemoryBytes
	b.active--
	if b.closed && b.active == 0 {
		close(b.drained)
	} else {
		b.dispatch()
	}
}
func (l *Lease) Release() {
	if l == nil || l.budget == nil {
		return
	}
	b := l.budget
	b.mu.Lock()
	defer b.mu.Unlock()
	b.release(l)
}

// Shrink atomically lowers a lease without releasing/reacquiring its retained
// dimensions. Useful after model loading: drop loading CPU slots, retain resident
// memory. Increasing demand or shrinking to zero is rejected; use Release for zero.
func (l *Lease) Shrink(r Resources) error {
	if l == nil || l.budget == nil || !valid(r) {
		return ErrInvalid
	}
	b := l.budget
	b.mu.Lock()
	defer b.mu.Unlock()
	if l.released {
		return ErrClosed
	}
	if r.CPUSlots > l.need.CPUSlots || r.MemoryBytes > l.need.MemoryBytes {
		return ErrInvalid
	}
	b.used.CPUSlots -= l.need.CPUSlots - r.CPUSlots
	b.used.MemoryBytes -= l.need.MemoryBytes - r.MemoryBytes
	l.need = r
	b.dispatch()
	return nil
}

// Admission binds a validated fixed reservation for callback-based owners such
// as speechjob.Queue. Construction reserves nothing and starts no goroutine.
func (b *Budget) Admission(r Resources) (func(context.Context) (func(), error), error) {
	if !valid(r) {
		return nil, ErrInvalid
	}
	if r.CPUSlots > b.cfg.Capacity.CPUSlots || r.MemoryBytes > b.cfg.Capacity.MemoryBytes {
		return nil, ErrCapacity
	}
	return func(ctx context.Context) (func(), error) {
		l, e := b.Acquire(ctx, r)
		if e != nil {
			return nil, e
		}
		return l.Release, nil
	}, nil
}

func (b *Budget) Snapshot() Snapshot {
	b.mu.Lock()
	defer b.mu.Unlock()
	return Snapshot{b.cfg.Capacity, b.used, b.active, len(b.waiting), b.closed}
}

// Close refuses future admission and wakes pending callers. It NEVER releases
// active leases or cancels their work. Safe to call more than once.
func (b *Budget) Close() {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closed {
		return
	}
	b.closed = true
	for i, w := range b.waiting {
		w.err = ErrClosed
		close(w.ready)
		b.waiting[i] = nil
	}
	b.waiting = b.waiting[:0]
	if b.active == 0 {
		close(b.drained)
	}
}

// Shutdown closes admission and waits for explicit release of all active leases.
// Timeout is not permission to free in-use resources; retry after owners drain.
func (b *Budget) Shutdown(ctx context.Context) error {
	if ctx == nil {
		return ErrInvalid
	}
	b.Close()
	select {
	case <-b.drained:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}
