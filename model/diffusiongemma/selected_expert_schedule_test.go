package diffusiongemma

import "testing"

func TestSelectedExpertSchedulePositions8TopK4(t *testing.T) {
	positions, topK, experts := 8, 4, 16
	ids := make([]int, positions*topK)
	vals := make([]float32, positions*topK)
	for pos := 0; pos < positions; pos++ {
		for slot := 0; slot < topK; slot++ {
			ids[pos*topK+slot] = (pos*3 + slot*5) % experts
			vals[pos*topK+slot] = float32(slot+1) / 10
		}
	}
	items, err := FlattenSelectedExperts(ids, vals, positions, topK, experts)
	if err != nil {
		t.Fatal(err)
	}
	arr := BuildSelectedExpertWorkArrays(items)
	grouped, err := BuildSelectedExpertGroupedWork(arr, experts)
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
	if len(items) != positions*topK || len(ga.WorkPositions) != positions*topK {
		t.Fatalf("work len=%d grouped=%d want %d", len(items), len(ga.WorkPositions), positions*topK)
	}
	// Active experts are deterministic ascending IDs with at least one work item.
	wantActive := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
	if len(ga.ActiveExperts) != len(wantActive) {
		t.Fatalf("active=%v", ga.ActiveExperts)
	}
	for i := range wantActive {
		if ga.ActiveExperts[i] != wantActive[i] {
			t.Fatalf("active=%v want=%v", ga.ActiveExperts, wantActive)
		}
	}
	if ga.Offsets[0] != 0 || ga.Offsets[len(ga.Offsets)-1] != positions*topK {
		t.Fatalf("offsets endpoints=%v", ga.Offsets)
	}
	// Every grouped work item must map back to the original flat top-k tables.
	for gi, pos := range ga.WorkPositions {
		slot := ga.WorkSlots[gi]
		expert := ga.ActiveExperts[groupForOffset(ga.Offsets, gi)]
		flat := pos*topK + slot
		if ids[flat] != expert || vals[flat] != ga.WorkWeights[gi] {
			t.Fatalf("grouped[%d] pos=%d slot=%d expert=%d weight=%g original expert=%d weight=%g", gi, pos, slot, expert, ga.WorkWeights[gi], ids[flat], vals[flat])
		}
	}
}

func groupForOffset(offsets []int, idx int) int {
	for g := 0; g+1 < len(offsets); g++ {
		if idx >= offsets[g] && idx < offsets[g+1] {
			return g
		}
	}
	return -1
}
