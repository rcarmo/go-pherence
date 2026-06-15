package diffusiongemma

import (
	"testing"

	gpu "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

func TestSelectedExpertGroupedArraysGPUBuffersUpload(t *testing.T) {
	if !gpu.SgemmReady() {
		t.Skip("CUDA not available")
	}
	arr := BuildSelectedExpertWorkArrays([]SelectedExpertWorkItem{{Position: 0, Expert: 2, Slot: 0, Weight: 0.5}, {Position: 1, Expert: 1, Slot: 0, Weight: 0.25}, {Position: 2, Expert: 2, Slot: 1, Weight: 0.75}})
	g, err := BuildSelectedExpertGroupedWork(arr, 4)
	if err != nil {
		t.Fatal(err)
	}
	ga, err := BuildSelectedExpertGroupedArrays(arr, g)
	if err != nil {
		t.Fatal(err)
	}
	var bufs SelectedExpertGroupedArraysGPUBuffers
	defer bufs.Free()
	if err := bufs.Upload(ga); err != nil {
		t.Fatal(err)
	}
	if bufs.WorkCapacity < len(ga.WorkPositions) || bufs.GroupCapacity < len(ga.ActiveExperts) || bufs.WorkPositions == nil || bufs.WorkWeights == nil || bufs.WorkDownScales == nil || bufs.EffectiveWeights == nil || bufs.WorkSlots == nil || bufs.WorkActive == nil || bufs.ActiveExperts == nil || bufs.Offsets == nil {
		t.Fatalf("bad grouped arrays gpu buffers: %+v", bufs)
	}
	eff := make([]float32, len(ga.WorkWeights))
	if err := bufs.EffectiveWeights.Download(eff); err != nil {
		t.Fatal(err)
	}
	for i := range eff {
		if eff[i] != ga.WorkWeights[i]*ga.WorkDownScales[i] {
			t.Fatalf("effective[%d]=%g want %g", i, eff[i], ga.WorkWeights[i]*ga.WorkDownScales[i])
		}
	}
	capWork, capGroup := bufs.WorkCapacity, bufs.GroupCapacity
	if err := bufs.Upload(ga); err != nil {
		t.Fatal(err)
	}
	if bufs.WorkCapacity != capWork || bufs.GroupCapacity != capGroup {
		t.Fatal("capacity changed")
	}
}
