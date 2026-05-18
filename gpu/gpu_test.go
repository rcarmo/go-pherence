package gpu

import "testing"

func TestGPUInit(t *testing.T) {
	if !Available() {
		t.Skip("GPU not available (no libcuda.so.1)")
	}
	t.Logf("GPU: %s (%d SMs)", DeviceName(), SMCount())
}
