package diffusiongemma

import "testing"

func TestBuildSelectedExpertWorkArrays(t *testing.T) {
	items := []SelectedExpertWorkItem{{Position: 2, Expert: 5, Slot: 1, Weight: 0.25}, {Position: 0, Expert: 1, Slot: 0, Weight: 0.75}}
	arr := BuildSelectedExpertWorkArrays(items)
	if err := arr.Validate(); err != nil {
		t.Fatal(err)
	}
	if arr.Len() != 2 || arr.Positions[0] != 2 || arr.Experts[0] != 5 || arr.Slots[0] != 1 || arr.Weights[0] != 0.25 || arr.Positions[1] != 0 || arr.Experts[1] != 1 || arr.Slots[1] != 0 || arr.Weights[1] != 0.75 {
		t.Fatalf("bad arrays: %+v", arr)
	}
	if arr.PositionsU[0] != 2 || arr.ExpertsU[0] != 5 || arr.SlotsU[0] != 1 || arr.PositionsU[1] != 0 || arr.ExpertsU[1] != 1 || arr.SlotsU[1] != 0 {
		t.Fatalf("bad uint32 arrays: %+v", arr)
	}
}

func TestSelectedExpertWorkArraysValidateRejectsMismatch(t *testing.T) {
	arr := SelectedExpertWorkArrays{Positions: []int{0}, Experts: nil, Slots: []int{0}, Weights: []float32{1}, PositionsU: []uint32{0}, ExpertsU: []uint32{0}, SlotsU: []uint32{0}}
	if err := arr.Validate(); err == nil {
		t.Fatal("expected length mismatch")
	}
}
