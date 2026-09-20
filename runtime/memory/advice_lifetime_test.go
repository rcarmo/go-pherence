package memory

import (
	"sync"
	"syscall"
	"testing"
	"time"
)

func TestDetachWaitsForAdviceAndDisablesFutureCalls(t *testing.T) {
	a, p := mappedAdvisor(t)
	entered, release := make(chan struct{}), make(chan struct{})
	var once sync.Once
	defer once.Do(func() { close(release) })
	a.advise = func([]byte, int) error { close(entered); <-release; return nil }
	prefetch := make(chan error, 1)
	go func() { prefetch <- a.Prefetch(0, p) }()
	<-entered
	detached := make(chan struct{})
	go func() { a.Detach(); close(detached) }()
	select {
	case <-detached:
		t.Fatal("Detach returned while advice active")
	case <-time.After(20 * time.Millisecond):
	}
	once.Do(func() { close(release) })
	if err := <-prefetch; err != nil {
		t.Fatal(err)
	}
	<-detached
	a.advise = func([]byte, int) error { t.Fatal("advice after detach"); return nil }
	a.Detach()
	a.Touch(0, p)
	if err := a.Prefetch(0, p); err != nil {
		t.Fatal(err)
	}
	if err := a.Evict(0, p); err != nil {
		t.Fatal(err)
	}
	if n, err := a.EvictCold(time.Now().UnixNano()); n != 0 || err != nil {
		t.Fatal(n, err)
	}
	if n, hot, _ := a.Stats(); n != 0 || hot != 0 {
		t.Fatal(n, hot)
	}
	var nilAdvisor *MmapAdvisor
	nilAdvisor.Detach()
}
func TestColdAdviceSerializesTouchAndAccounting(t *testing.T) {
	a, p := mappedAdvisor(t)
	a.Touch(0, p)
	cutoff := time.Now().UnixNano() + 1
	entered, release := make(chan struct{}), make(chan struct{})
	var once sync.Once
	defer once.Do(func() { close(release) })
	a.advise = func(_ []byte, advice int) error {
		if advice != syscall.MADV_DONTNEED {
			t.Error(advice)
		}
		close(entered)
		<-release
		return nil
	}
	evicted := make(chan error, 1)
	go func() { _, err := a.EvictCold(cutoff); evicted <- err }()
	<-entered
	touched := make(chan struct{})
	go func() { a.Touch(0, p); close(touched) }()
	select {
	case <-touched:
		t.Fatal("Touch completed during eviction syscall")
	case <-time.After(20 * time.Millisecond):
	}
	once.Do(func() { close(release) })
	if err := <-evicted; err != nil {
		t.Fatal(err)
	}
	<-touched
	if _, hot, _ := a.Stats(); hot != p {
		t.Fatal("new touch lost", hot)
	}
}
