package diffusiongemma

import "testing"

func TestBuildSelectedExpertGroupedWork(t *testing.T) {
	items := []SelectedExpertWorkItem{
		{Position: 0, Expert: 2, Slot: 0, Weight: 0.5},
		{Position: 0, Expert: 4, Slot: 1, Weight: 0.25},
		{Position: 1, Expert: 2, Slot: 0, Weight: 0.75},
		{Position: 1, Expert: 0, Slot: 1, Weight: 0.125},
	}
	arr := BuildSelectedExpertWorkArrays(items)
	g, err := BuildSelectedExpertGroupedWork(arr, 5)
	if err != nil {
		t.Fatal(err)
	}
	if err := g.Validate(arr.Len()); err != nil {
		t.Fatal(err)
	}
	wantActive := []int{0, 2, 4}
	wantOffsets := []int{0, 1, 3, 4}
	wantOrder := []int{3, 0, 2, 1}
	for i := range wantActive {
		if g.ActiveExperts[i] != wantActive[i] {
			t.Fatalf("active=%v want %v", g.ActiveExperts, wantActive)
		}
	}
	for i := range wantOffsets {
		if g.Offsets[i] != wantOffsets[i] {
			t.Fatalf("offsets=%v want %v", g.Offsets, wantOffsets)
		}
	}
	for i := range wantOrder {
		if g.WorkOrder[i] != wantOrder[i] {
			t.Fatalf("order=%v want %v", g.WorkOrder, wantOrder)
		}
	}
}

func TestSelectedExpertGroupedScheduleMatchesKnownTopK(t *testing.T) {
	ids := []int{0, 7, 13, 13, 7, 1}
	vals := []float32{0.55, 0.30, 0.15, 0.50, 0.25, 0.25}
	items, err := FlattenSelectedExperts(ids, vals, 2, 3, 128)
	if err != nil {
		t.Fatal(err)
	}
	arr := BuildSelectedExpertWorkArrays(items)
	grouped, err := BuildSelectedExpertGroupedWork(arr, 128)
	if err != nil {
		t.Fatal(err)
	}
	ga, err := BuildSelectedExpertGroupedArrays(arr, grouped)
	if err != nil {
		t.Fatal(err)
	}
	if err := ga.Validate(); err != nil {
		t.Fatal(err)
	}
	wantActive := []int{0, 1, 7, 13}
	wantOffsets := []int{0, 1, 2, 4, 6}
	wantPositions := []int{0, 1, 0, 1, 0, 1}
	wantWeights := []float32{0.55, 0.25, 0.30, 0.25, 0.15, 0.50}
	for i := range wantActive {
		if ga.ActiveExperts[i] != wantActive[i] {
			t.Fatalf("active=%v want %v", ga.ActiveExperts, wantActive)
		}
	}
	for i := range wantOffsets {
		if ga.Offsets[i] != wantOffsets[i] {
			t.Fatalf("offsets=%v want %v", ga.Offsets, wantOffsets)
		}
	}
	for i := range wantPositions {
		if ga.WorkPositions[i] != wantPositions[i] || ga.WorkWeights[i] != wantWeights[i] {
			t.Fatalf("positions=%v weights=%v", ga.WorkPositions, ga.WorkWeights)
		}
	}
}

func TestSelectedExpertGroupedWorkValidateRejectsBadOffsets(t *testing.T) {
	g := SelectedExpertGroupedWork{WorkOrder: []int{0}, ActiveExperts: []int{1}, Offsets: []int{1, 1}}
	if err := g.Validate(1); err == nil {
		t.Fatal("expected invalid offsets")
	}
}
