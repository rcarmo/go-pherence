package nvidia

import (
	"os"
	"testing"
)

// Each init owner is a short-lived goroutine. Init must balance LockOSThread;
// otherwise exiting that goroutine terminates its OS thread while CUDA still
// has thread-local state and can crash after the test has printed PASS.
func TestCUDAInitFromShortLivedGoroutine(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_CUDA_LIFECYCLE") != "1" || os.Getenv("GO_PHERENCE_DISABLE_NVIDIA") != "" {
		t.Skip("opt-in CUDA lifecycle")
	}
	for i := 0; i < 3; i++ {
		done := make(chan bool, 1)
		go func() { done <- Available() }()
		if !<-done {
			t.Fatal("CUDA unavailable")
		}
		if !SgemmReady() {
			t.Fatal("kernel load failed")
		}
		b, err := Malloc(16)
		if err != nil {
			t.Fatal(err)
		}
		if err = b.Upload(make([]float32, 16)); err != nil {
			t.Fatal(err)
		}
		b.Free()
		Shutdown()
	}
}
