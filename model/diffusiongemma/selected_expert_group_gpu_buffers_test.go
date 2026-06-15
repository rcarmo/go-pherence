package diffusiongemma

import (
	"testing"

	gpu "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

func TestSelectedExpertGroupedWorkGPUBuffersUpload(t *testing.T) {
	if !gpu.SgemmReady() {
		t.Skip("CUDA not available")
	}
	arr := BuildSelectedExpertWorkArrays([]SelectedExpertWorkItem{{Position: 0, Expert: 2, Slot: 0, Weight: 0.5}, {Position: 1, Expert: 1, Slot: 0, Weight: 0.25}, {Position: 2, Expert: 2, Slot: 1, Weight: 0.75}})
	g, err := BuildSelectedExpertGroupedWork(arr, 4)
	if err != nil {
		t.Fatal(err)
	}
	var bufs SelectedExpertGroupedWorkGPUBuffers
	defer bufs.Free()
	if err := bufs.Upload(g, arr.Len()); err != nil {
		t.Fatal(err)
	}
	if bufs.WorkCapacity < arr.Len() || bufs.GroupCapacity < len(g.ActiveExperts) || bufs.WorkOrder == nil || bufs.ActiveExperts == nil || bufs.Offsets == nil {
		t.Fatalf("bad grouped gpu buffers: %+v", bufs)
	}
	capWork, capGroup := bufs.WorkCapacity, bufs.GroupCapacity
	if err := bufs.Upload(g, arr.Len()); err != nil {
		t.Fatal(err)
	}
	if bufs.WorkCapacity != capWork || bufs.GroupCapacity != capGroup {
		t.Fatalf("capacity changed")
	}
}
