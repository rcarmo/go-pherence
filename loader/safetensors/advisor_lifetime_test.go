package safetensors

import (
	"sync"
	"testing"
)

func TestRetainedAdvisorDetachedOnFileClose(t *testing.T) {
	p := writeTestSafetensors(t, `{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`, []byte{0, 0, 128, 63})
	f, err := Open(p)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	a := f.Advisor // Capture while owner is open; do not race reads of File fields.
	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				_ = a.Prefetch(0, 1)
				a.Touch(0, 1)
				_ = a.Evict(0, 1)
			}
		}()
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	wg.Wait()
	// No native calls can now target an unmapped address, even with a retained handle.
	beforePrefetch, beforeEvict := a.TotalPrefetches.Load(), a.TotalEvictions.Load()
	if err := a.Prefetch(0, 1); err != nil {
		t.Fatal(err)
	}
	if err := a.Evict(0, 1); err != nil {
		t.Fatal(err)
	}
	a.Touch(0, 1)
	if a.TotalPrefetches.Load() != beforePrefetch || a.TotalEvictions.Load() != beforeEvict {
		t.Fatal("detached advice counted")
	}
	if n, hot, _ := a.Stats(); n != 0 || hot != 0 {
		t.Fatal("detached state", n, hot)
	}
}
