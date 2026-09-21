package ime2

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

type lifecyclePool interface {
	Run(func(int, int))
	Close()
}

// No pinning or IME execution: this checks the real channel/condition dispatch
// protocols with ordinary Go callbacks, not a scalar stand-in for native GEMM.
func TestPoolLifecycle(t *testing.T) {
	factories := map[string]func(int) lifecyclePool{
		"channels":  func(n int) lifecyclePool { return newWorkerPool(n, nil) },
		"condition": func(n int) lifecyclePool { return newCondPool(n, nil) },
	}
	for name, newPool := range factories {
		t.Run(name, func(t *testing.T) {
			p := newPool(3)
			defer p.Close()
			const submissions = 24
			var group sync.WaitGroup
			var counts [submissions][3]atomic.Int32
			for job := 0; job < submissions; job++ {
				group.Add(1)
				go func() {
					defer group.Done()
					p.Run(func(id, n int) {
						if n != 3 || id < 0 || id >= 3 {
							panic("worker geometry")
						}
						counts[job][id].Add(1)
					})
				}()
			}
			group.Wait()
			for job := range counts {
				for id := range counts[job] {
					if counts[job][id].Load() != 1 {
						t.Fatalf("job %d worker %d count=%d", job, id, counts[job][id].Load())
					}
				}
			}
			entered := make(chan struct{}, 3)
			release := make(chan struct{})
			runDone := make(chan struct{})
			go func() { p.Run(func(int, int) { entered <- struct{}{}; <-release }); close(runDone) }()
			for i := 0; i < 3; i++ {
				<-entered
			}
			closeDone := make(chan struct{})
			go func() { p.Close(); close(closeDone) }()
			select {
			case <-closeDone:
				t.Error("Close returned before callback completion")
			case <-time.After(10 * time.Millisecond):
			}
			close(release)
			select {
			case <-closeDone:
			case <-time.After(3 * time.Second):
				t.Fatal("Close did not drain/join")
			}
			<-runDone
			p.Close()
			p.Run(func(int, int) { t.Error("Run after Close executed") })
			p.Run(nil)
		})
		t.Run(name+"/zero", func(t *testing.T) {
			p := newPool(0)
			p.Run(nil)
			p.Run(func(int, int) { t.Error("zero worker ran") })
			p.Close()
			p.Close()
		})
	}
}
