package diffusiongemma

import "testing"

func TestBuildSelectedExpertGroupedArrays(t *testing.T) {
	items := []SelectedExpertWorkItem{
		{Position: 0, Expert: 2, Slot: 0, Weight: 0.5},
		{Position: 0, Expert: 4, Slot: 1, Weight: 0.25},
		{Position: 1, Expert: 2, Slot: 0, Weight: 0.75},
		{Position: 1, Expert: 0, Slot: 1, Weight: 0.125},
	}
	arr := BuildSelectedExpertWorkArrays(items)
	grouped, err := BuildSelectedExpertGroupedWork(arr, 5)
	if err != nil {
		t.Fatal(err)
	}
	ga, err := BuildSelectedExpertGroupedArrays(arr, grouped)
	if err != nil {
		t.Fatal(err)
	}
	if err := ga.ApplyDownScalesByExpert([]float32{1.5, 1, 2, 1, 0.5}); err != nil {
		t.Fatal(err)
	}
	if err := ga.Validate(); err != nil {
		t.Fatal(err)
	}
	wantPos := []int{1, 0, 1, 0}
	wantWeights := []float32{0.125, 0.5, 0.75, 0.25}
	wantScales := []float32{1.5, 2, 2, 0.5}
	wantActive := []int{0, 2, 4}
	wantOffsets := []int{0, 1, 3, 4}
	for i := range wantPos {
		if ga.WorkPositions[i] != wantPos[i] || ga.WorkPositionsU[i] != uint32(wantPos[i]) {
			t.Fatalf("positions=%v/%v", ga.WorkPositions, ga.WorkPositionsU)
		}
	}
	for i := range wantWeights {
		if ga.WorkWeights[i] != wantWeights[i] {
			t.Fatalf("weights=%v", ga.WorkWeights)
		}
	}
	for i := range wantScales {
		if ga.WorkDownScales[i] != wantScales[i] {
			t.Fatalf("down scales=%v", ga.WorkDownScales)
		}
	}
	for i := range wantActive {
		if ga.ActiveExperts[i] != wantActive[i] || ga.ActiveExpertsU[i] != uint32(wantActive[i]) {
			t.Fatalf("active=%v/%v", ga.ActiveExperts, ga.ActiveExpertsU)
		}
	}
	for i := range wantOffsets {
		if ga.Offsets[i] != wantOffsets[i] || ga.OffsetsU[i] != uint32(wantOffsets[i]) {
			t.Fatalf("offsets=%v/%v", ga.Offsets, ga.OffsetsU)
		}
	}
	wantActiveForWork := []int{0, 1, 1, 2}
	for i := range wantActiveForWork {
		if ga.WorkActive[i] != wantActiveForWork[i] || ga.WorkActiveU[i] != uint32(wantActiveForWork[i]) {
			t.Fatalf("work active=%v/%v", ga.WorkActive, ga.WorkActiveU)
		}
	}
}

func TestSelectedExpertGroupedArraysValidateRejectsMismatch(t *testing.T) {
	ga := SelectedExpertGroupedArrays{WorkPositions: []int{0}, WorkPositionsU: []uint32{0}, WorkWeights: []float32{1}, WorkDownScales: []float32{1}, WorkSlots: []int{0}, WorkSlotsU: []uint32{0}, WorkActive: []int{0}, WorkActiveU: []uint32{0}, ActiveExperts: []int{1}, ActiveExpertsU: []uint32{1}, Offsets: []int{0}, OffsetsU: []uint32{0}}
	if err := ga.Validate(); err == nil {
		t.Fatal("expected mismatch")
	}
}
