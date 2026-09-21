//go:build linux && riscv64

package aipool

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// TestMain requires GO_PHERENCE_TEST_K3 and /proc/set_ai_thread. This test is
// compile-only off-device; it exercises real registration/affinity/TCM ownership.
func TestAIWorkerPoolDrainsBeforeClose(t *testing.T) {
	p := NewAIWorkerPool(2)
	defer p.Close()
	var wg sync.WaitGroup
	var calls atomic.Int32
	for i := 0; i < 12; i++ {
		wg.Add(1)
		go func() { defer wg.Done(); p.Run(func(int, int) { calls.Add(1) }) }()
	}
	wg.Wait()
	if calls.Load() != 24 {
		t.Fatal(calls.Load())
	}
	entered := make(chan struct{}, 2)
	release := make(chan struct{})
	done := make(chan struct{})
	go func() {
		p.Run(func(id, n int) {
			entered <- struct{}{}
			<-release
			if len(p.TcmSlices) > id && len(p.TcmSlices[id]) > 0 {
				p.TcmSlices[id][0] = byte(id)
			}
		})
		close(done)
	}()
	<-entered
	<-entered
	closed := make(chan struct{})
	go func() { p.Close(); close(closed) }()
	select {
	case <-closed:
		t.Error("unmapped TCM with active workers")
	case <-time.After(10 * time.Millisecond):
	}
	close(release)
	select {
	case <-closed:
	case <-time.After(5 * time.Second):
		t.Fatal("workers did not drain")
	}
	<-done
	if p.TcmSlices != nil {
		t.Fatal("closed pool retained TCM views")
	}
	p.Close()
	p.Run(func(int, int) { t.Error("closed pool executed") })
}
