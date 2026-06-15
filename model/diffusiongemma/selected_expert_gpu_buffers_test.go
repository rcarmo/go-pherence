package diffusiongemma

import (
	"testing"

	gpu "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

func TestSelectedExpertWorkGPUBuffersUpload(t *testing.T) {
	if !gpu.SgemmReady() {
		t.Skip("CUDA not available")
	}
	arr := BuildSelectedExpertWorkArrays([]SelectedExpertWorkItem{{Position: 1, Expert: 2, Slot: 0, Weight: 0.5}, {Position: 3, Expert: 4, Slot: 1, Weight: 0.25}})
	var bufs SelectedExpertWorkGPUBuffers
	defer bufs.Free()
	if err := bufs.Upload(arr); err != nil {
		t.Fatal(err)
	}
	if bufs.Capacity < arr.Len() || bufs.Positions == nil || bufs.Experts == nil || bufs.Slots == nil || bufs.Weights == nil {
		t.Fatalf("bad gpu buffers after upload: %+v", bufs)
	}
	cap0 := bufs.Capacity
	if err := bufs.Upload(arr); err != nil {
		t.Fatal(err)
	}
	if bufs.Capacity != cap0 {
		t.Fatalf("capacity changed on same-size upload: %d -> %d", cap0, bufs.Capacity)
	}
}
