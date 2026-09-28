// Package contextmutex provides a reusable context-aware mutex.
package contextmutex

import (
	"context"
	"errors"
	"sync"
)

// ErrNilContext is returned by [Mutex.LockContext] when ctx is nil.
var ErrNilContext = errors.New("contextmutex: nil context")

type noCopy struct{}

func (*noCopy) Lock()   {}
func (*noCopy) Unlock() {}

// Mutex is a reusable mutual exclusion primitive with context-aware locking.
//
// The zero value is ready for use. Like [sync.Mutex], a Mutex must not be
// copied after first use. Mutex makes no fairness or starvation guarantees.
type Mutex struct {
	noCopy noCopy
	once   sync.Once
	sem    chan struct{}
}

func (m *Mutex) init() {
	m.once.Do(func() {
		m.sem = make(chan struct{}, 1)
		m.sem <- struct{}{}
	})
}

// Lock locks m. It is equivalent to calling [Mutex.LockContext] with
// [context.Background].
func (m *Mutex) Lock() {
	_ = m.LockContext(context.Background())
}

// LockContext locks m or returns ctx's error if ctx is canceled first. If ctx
// is nil, LockContext returns [ErrNilContext]. A canceled context never
// acquires the lock.
func (m *Mutex) LockContext(ctx context.Context) error {
	if ctx == nil {
		return ErrNilContext
	}
	m.init()
	if err := ctx.Err(); err != nil {
		return err
	}
	select {
	case <-m.sem:
		if err := ctx.Err(); err != nil {
			m.sem <- struct{}{}
			return err
		}
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// Unlock unlocks m. It panics if m is not locked.
func (m *Mutex) Unlock() {
	m.init()
	select {
	case m.sem <- struct{}{}:
		return
	default:
		panic("contextmutex: unlock of unlocked mutex")
	}
}
