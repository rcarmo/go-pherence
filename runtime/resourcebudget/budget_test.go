package resourcebudget

import (
	"context"
	"errors"
	"math"
	"sync"
	"testing"
	"time"
)

func makeBudget(t *testing.T, c Config) *Budget {
	t.Helper()
	b, e := New(c)
	if e != nil {
		t.Fatal(e)
	}
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		defer cancel()
		if e := b.Shutdown(ctx); e != nil {
			t.Error(e, b.Snapshot())
		}
	})
	return b
}
func acquire(t *testing.T, b *Budget, r Resources) *Lease {
	t.Helper()
	l, e := b.Acquire(context.Background(), r)
	if e != nil {
		t.Fatal(e)
	}
	t.Cleanup(l.Release)
	return l
}
func waitCount(t *testing.T, b *Budget, n int) {
	t.Helper()
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		if b.Snapshot().Waiting == n {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatal("waiting count", b.Snapshot(), n)
}
func TestAtomicDimensionsAndShrink(t *testing.T) {
	b := makeBudget(t, Config{Resources{4, 100}, 4, 4})
	resident := acquire(t, b, Resources{4, 60})
	if e := resident.Shrink(Resources{0, 60}); e != nil {
		t.Fatal(e)
	}
	work := acquire(t, b, Resources{2, 30})
	s := b.Snapshot()
	if s.Used != (Resources{2, 90}) || s.Active != 2 {
		t.Fatal(s)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		l, e := b.Acquire(ctx, Resources{1, 20})
		if l != nil {
			l.Release()
		}
		done <- e
	}()
	waitCount(t, b, 1)
	if b.Snapshot().Used != s.Used {
		t.Fatal("partial reservation")
	}
	cancel()
	if e := <-done; !errors.Is(e, context.Canceled) {
		t.Fatal(e)
	}
	if e := resident.Shrink(Resources{1, 60}); !errors.Is(e, ErrInvalid) {
		t.Fatal("growth accepted", e)
	}
	if e := resident.Shrink(Resources{}); !errors.Is(e, ErrInvalid) {
		t.Fatal(e)
	}
	work.Release()
	work.Release()
	resident.Release()
	if b.Snapshot().Used != (Resources{}) {
		t.Fatal(b.Snapshot())
	}
	if e := work.Shrink(Resources{1, 1}); !errors.Is(e, ErrClosed) {
		t.Fatal(e)
	}
}
func TestFIFOHeadBlockingAndCancellation(t *testing.T) {
	b := makeBudget(t, Config{Resources{4, 100}, 4, 4})
	hold := acquire(t, b, Resources{3, 30})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	first := make(chan error, 1)
	small := make(chan *Lease, 1)
	go func() {
		l, e := b.Acquire(ctx, Resources{2, 10})
		if l != nil {
			l.Release()
		}
		first <- e
	}()
	waitCount(t, b, 1)
	go func() {
		l, e := b.Acquire(context.Background(), Resources{1, 10})
		if e != nil {
			t.Error(e)
		}
		small <- l
	}()
	waitCount(t, b, 2)
	select {
	case l := <-small:
		l.Release()
		t.Fatal("small request bypassed head")
	default:
	}
	cancel()
	if e := <-first; !errors.Is(e, context.Canceled) {
		t.Fatal(e)
	}
	select {
	case l := <-small:
		l.Release()
	case <-time.After(time.Second):
		t.Fatal("cancel did not wake next")
	}
	hold.Release()
}
func TestWaitingActiveAndOverflowBounds(t *testing.T) {
	b := makeBudget(t, Config{Resources{math.MaxInt, math.MaxInt64}, 1, 1})
	hold := acquire(t, b, Resources{math.MaxInt - 1, math.MaxInt64 - 1})
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		l, e := b.Acquire(ctx, Resources{1, 1})
		if l != nil {
			l.Release()
		}
		done <- e
	}()
	waitCount(t, b, 1)
	if l, e := b.Acquire(context.Background(), Resources{1, 1}); l != nil || !errors.Is(e, ErrBusy) {
		t.Fatal(l, e)
	}
	cancel()
	<-done
	hold.Release()
	l := acquire(t, b, Resources{math.MaxInt, math.MaxInt64})
	if b.Snapshot().Used != (Resources{math.MaxInt, math.MaxInt64}) {
		t.Fatal(b.Snapshot())
	}
	l.Release()
	for _, r := range []Resources{{-1, 1}, {1, -1}, {0, 0}} {
		if _, e := b.Acquire(context.Background(), r); !errors.Is(e, ErrInvalid) {
			t.Fatal(r, e)
		}
	}
	small := makeBudget(t, Config{Resources{2, 10}, 1, 0})
	if _, e := small.Acquire(context.Background(), Resources{3, 1}); !errors.Is(e, ErrCapacity) {
		t.Fatal(e)
	}
	if _, e := small.Acquire(context.Background(), Resources{1, 11}); !errors.Is(e, ErrCapacity) {
		t.Fatal(e)
	}
	h := acquire(t, small, Resources{1, 1})
	if _, e := small.Acquire(context.Background(), Resources{1, 1}); !errors.Is(e, ErrBusy) {
		t.Fatal(e)
	}
	h.Release()
}
func TestContextDoesNotRevokeGrantedLeaseAndShutdown(t *testing.T) {
	b := makeBudget(t, Config{Resources{2, 10}, 2, 4})
	ctx, cancel := context.WithCancel(context.Background())
	l, e := b.Acquire(ctx, Resources{2, 10})
	if e != nil {
		t.Fatal(e)
	}
	defer l.Release()
	cancel()
	if s := b.Snapshot(); s.Active != 1 || s.Used.MemoryBytes != 10 {
		t.Fatal("revoked live resource", s)
	}
	pending := make(chan error, 1)
	go func() { _, e := b.Acquire(context.Background(), Resources{1, 1}); pending <- e }()
	waitCount(t, b, 1)
	timeout, stop := context.WithTimeout(context.Background(), time.Millisecond)
	defer stop()
	if e = b.Shutdown(timeout); !errors.Is(e, context.DeadlineExceeded) {
		t.Fatal(e)
	}
	if e = <-pending; !errors.Is(e, ErrClosed) {
		t.Fatal(e)
	}
	if _, e = b.Acquire(context.Background(), Resources{1, 1}); !errors.Is(e, ErrClosed) {
		t.Fatal(e)
	}
	if s := b.Snapshot(); s.Active != 1 || s.Waiting != 0 || !s.Closed {
		t.Fatal(s)
	}
	l.Release()
	if e = b.Shutdown(context.Background()); e != nil {
		t.Fatal(e)
	}
}
func TestConcurrentGrantCancelRelease(t *testing.T) {
	b := makeBudget(t, Config{Resources{4, 40}, 4, 64})
	var wg sync.WaitGroup
	for range 64 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 20 {
				ctx, cancel := context.WithCancel(context.Background())
				go cancel()
				l, e := b.Acquire(ctx, Resources{1, 10})
				if l != nil {
					l.Release()
					l.Release()
				} else if !errors.Is(e, context.Canceled) && !errors.Is(e, ErrBusy) {
					t.Error(e)
				}
				s := b.Snapshot()
				if s.Used.CPUSlots > 4 || s.Used.MemoryBytes > 40 || s.Used.CPUSlots < 0 || s.Used.MemoryBytes < 0 {
					t.Error(s)
				}
			}
		}()
	}
	wg.Wait()
	if s := b.Snapshot(); s.Active != 0 || s.Waiting != 0 || s.Used != (Resources{}) {
		t.Fatal(s)
	}
}
func TestBudgetInvalidAndAllocationBound(t *testing.T) {
	for _, c := range []Config{{}, {Resources{1, 1}, 0, 1}, {Resources{1, 1}, 1, -1}, {Resources{1, 1}, 65537, 1}, {Resources{1, 1}, 1, 65537}} {
		if _, e := New(c); !errors.Is(e, ErrInvalid) {
			t.Fatal(c, e)
		}
	}
	b := makeBudget(t, Config{Resources{2, 10}, 2, 1})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, e := b.Acquire(ctx, Resources{1, 1}); !errors.Is(e, context.Canceled) {
		t.Fatal(e)
	}
	if _, e := b.Acquire(nil, Resources{1, 1}); !errors.Is(e, ErrInvalid) {
		t.Fatal(e)
	}
	n := testing.AllocsPerRun(100, func() {
		l, e := b.Acquire(context.Background(), Resources{1, 1})
		if e != nil {
			panic(e)
		}
		l.Release()
	})
	if n > 1 {
		t.Fatal("uncontended lease allocations", n)
	}
	snapshot := testing.AllocsPerRun(100, func() { _ = b.Snapshot() })
	if snapshot != 0 {
		t.Fatal(snapshot)
	}
	t.Logf("uncontended lease/release %.0f allocations; snapshot %.0f; no timing claim", n, snapshot)
}

func TestDispatchCompactsCancellationAndDropsContexts(t *testing.T) {
	b := makeBudget(t, Config{Resources{4, 100}, 4, 16})
	for i := 0; i < 16; i++ {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		b.waiting = append(b.waiting, &waiter{ctx: ctx, need: Resources{1, 1}, ready: make(chan struct{})})
	}
	b.mu.Lock()
	b.dispatch()
	b.mu.Unlock()
	if len(b.waiting) != 0 {
		t.Fatal("cancelled waiters retained")
	}
	for _, w := range b.waiting[:cap(b.waiting)] {
		if w != nil {
			t.Fatal("context retained beyond length")
		}
	}
	cb, e := b.Admission(Resources{1, 10})
	if e != nil {
		t.Fatal(e)
	}
	release, e := cb(context.Background())
	if e != nil {
		t.Fatal(e)
	}
	release()
	release()
	if _, e = b.Admission(Resources{}); !errors.Is(e, ErrInvalid) {
		t.Fatal(e)
	}
	if _, e = b.Admission(Resources{1, 101}); !errors.Is(e, ErrCapacity) {
		t.Fatal(e)
	}
	b.Close()
	if _, e = cb(context.Background()); !errors.Is(e, ErrClosed) {
		t.Fatal(e)
	}
	var zero Lease
	zero.Release()
	if e = zero.Shrink(Resources{1, 1}); !errors.Is(e, ErrInvalid) {
		t.Fatal(e)
	}
}
func TestActiveSlotWaitAfterCPUAndMemoryShrink(t *testing.T) {
	b := makeBudget(t, Config{Resources{4, 100}, 1, 2})
	hold := acquire(t, b, Resources{4, 100})
	if e := hold.Shrink(Resources{0, 1}); e != nil {
		t.Fatal(e)
	}
	got := make(chan *Lease, 1)
	go func() {
		l, e := b.Acquire(context.Background(), Resources{1, 1})
		if e != nil {
			t.Error(e)
		}
		got <- l
	}()
	waitCount(t, b, 1)
	hold.Release()
	l := <-got
	l.Release()
	if b.Snapshot().Active != 0 {
		t.Fatal(b.Snapshot())
	}
}
func TestMultipleReadyWaitersFIFO(t *testing.T) {
	b := makeBudget(t, Config{Resources{4, 100}, 4, 4})
	hold := acquire(t, b, Resources{4, 100})
	one := make(chan *Lease, 1)
	two := make(chan *Lease, 1)
	go func() {
		l, e := b.Acquire(context.Background(), Resources{4, 1})
		if e != nil {
			t.Error(e)
		}
		one <- l
	}()
	waitCount(t, b, 1)
	go func() {
		l, e := b.Acquire(context.Background(), Resources{1, 1})
		if e != nil {
			t.Error(e)
		}
		two <- l
	}()
	waitCount(t, b, 2)
	hold.Release()
	first := <-one
	select {
	case l := <-two:
		l.Release()
		t.Fatal("bypassed first")
	default:
	}
	first.Release()
	last := <-two
	last.Release()
}
