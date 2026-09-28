package contextmutex

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"
)

const testTimeout = time.Second

func TestMutexZeroValue(t *testing.T) {
	var m Mutex
	m.Lock()
	m.Unlock()
	if err := m.LockContext(context.Background()); err != nil {
		t.Fatal(err)
	}
	m.Unlock()
}

func TestMutexLockCompatibility(t *testing.T) {
	var m Mutex
	m.Lock()
	attempted := make(chan struct{})
	acquired := make(chan struct{})
	go func() {
		close(attempted)
		m.Lock()
		close(acquired)
		m.Unlock()
	}()
	<-attempted
	select {
	case <-acquired:
		t.Fatal("lock acquired while still held")
	default:
	}
	m.Unlock()
	select {
	case <-acquired:
	case <-time.After(testTimeout):
		t.Fatal("timed out waiting for Lock")
	}
}

func TestMutexLockContextNil(t *testing.T) {
	var m Mutex
	if err := m.LockContext(nil); !errors.Is(err, ErrNilContext) {
		t.Fatalf("LockContext(nil) = %v, want %v", err, ErrNilContext)
	}
	m.Lock()
	m.Unlock()
}

func TestMutexLockContextCanceled(t *testing.T) {
	var m Mutex
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := m.LockContext(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("LockContext(canceled) = %v, want %v", err, context.Canceled)
	}
	acquired := make(chan struct{})
	go func() {
		m.Lock()
		close(acquired)
		m.Unlock()
	}()
	select {
	case <-acquired:
	case <-time.After(testTimeout):
		t.Fatal("canceled context consumed the lock")
	}
}

func TestMutexLockContextCanceledWhileWaiting(t *testing.T) {
	var m Mutex
	m.Lock()
	ctx, cancel := context.WithCancel(context.Background())
	attempted := make(chan struct{})
	errCh := make(chan error, 1)
	go func() {
		close(attempted)
		errCh <- m.LockContext(ctx)
	}()
	<-attempted
	cancel()
	select {
	case err := <-errCh:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("LockContext(waiting canceled) = %v, want %v", err, context.Canceled)
		}
	case <-time.After(testTimeout):
		t.Fatal("timed out waiting for canceled waiter")
	}
	m.Unlock()
	acquired := make(chan struct{})
	go func() {
		m.Lock()
		close(acquired)
		m.Unlock()
	}()
	select {
	case <-acquired:
	case <-time.After(testTimeout):
		t.Fatal("lock leaked after canceled waiter")
	}
}

func TestMutexConcurrentIncrements(t *testing.T) {
	var (
		m  Mutex
		wg sync.WaitGroup
		n  int
	)
	const goroutines = 8
	const loops = 1000
	wg.Add(goroutines)
	for range goroutines {
		go func() {
			defer wg.Done()
			for range loops {
				m.Lock()
				n++
				m.Unlock()
			}
		}()
	}
	wg.Wait()
	if n != goroutines*loops {
		t.Fatalf("count = %d, want %d", n, goroutines*loops)
	}
}

func TestMutexWarmAllocations(t *testing.T) {
	t.Run("Lock", func(t *testing.T) {
		var m Mutex
		m.Lock()
		m.Unlock()
		allocs := testing.AllocsPerRun(1000, func() {
			m.Lock()
			m.Unlock()
		})
		if allocs != 0 {
			t.Fatalf("Lock allocs = %v, want 0", allocs)
		}
	})
	t.Run("LockContext", func(t *testing.T) {
		var m Mutex
		m.Lock()
		m.Unlock()
		allocs := testing.AllocsPerRun(1000, func() {
			if err := m.LockContext(context.Background()); err != nil {
				panic(err)
			}
			m.Unlock()
		})
		if allocs != 0 {
			t.Fatalf("LockContext allocs = %v, want 0", allocs)
		}
	})
}

func TestMutexDoubleUnlockPanics(t *testing.T) {
	var m Mutex
	m.Lock()
	m.Unlock()
	defer func() {
		if recover() == nil {
			t.Fatal("Unlock did not panic")
		}
	}()
	m.Unlock()
}

// Cancel exactly on the second Err poll to cover a cancellation racing with
// immediate acquisition, without relying on scheduling or wall-clock sleeps.
type cancelOnSecondPoll struct {
	context.Context
	cancel context.CancelFunc
	polls  int
}

func (c *cancelOnSecondPoll) Err() error {
	c.polls++
	if c.polls == 2 {
		c.cancel()
	}
	return c.Context.Err()
}
func TestMutexPostAcquireCancellation(t *testing.T) {
	var m Mutex
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if err := m.LockContext(&cancelOnSecondPoll{Context: ctx, cancel: cancel}); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if err := m.LockContext(context.Background()); err != nil {
		t.Fatal(err)
	}
	m.Unlock()
}
