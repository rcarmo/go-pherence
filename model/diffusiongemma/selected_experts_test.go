package diffusiongemma

import "testing"

func TestFlattenSelectedExperts(t *testing.T) {
	ids := []int{2, -1, 4, 1, 2, 0}
	vals := []float32{0.5, 0, 0.25, 0.75, 0.2, 0.8}
	items, err := FlattenSelectedExperts(ids, vals, 2, 3, 5)
	if err != nil {
		t.Fatal(err)
	}
	want := []SelectedExpertWorkItem{
		{Position: 0, Expert: 2, Slot: 0, Weight: 0.5},
		{Position: 0, Expert: 4, Slot: 2, Weight: 0.25},
		{Position: 1, Expert: 1, Slot: 0, Weight: 0.75},
		{Position: 1, Expert: 2, Slot: 1, Weight: 0.2},
		{Position: 1, Expert: 0, Slot: 2, Weight: 0.8},
	}
	if len(items) != len(want) {
		t.Fatalf("items len=%d want %d: %+v", len(items), len(want), items)
	}
	for i := range want {
		if items[i] != want[i] {
			t.Fatalf("items[%d]=+%v want %+v", i, items[i], want[i])
		}
	}
}

func TestFlattenSelectedExpertsRejectsBadExpert(t *testing.T) {
	_, err := FlattenSelectedExperts([]int{3}, []float32{1}, 1, 1, 3)
	if err == nil {
		t.Fatal("expected bad expert id error")
	}
}

func TestGroupSelectedExpertWork(t *testing.T) {
	items := []SelectedExpertWorkItem{{Position: 0, Expert: 2, Weight: 0.5}, {Position: 1, Expert: 0, Weight: 0.8}, {Position: 2, Expert: 2, Weight: 0.2}}
	groups, err := GroupSelectedExpertWork(items, 3)
	if err != nil {
		t.Fatal(err)
	}
	if len(groups[0]) != 1 || groups[0][0].Position != 1 || len(groups[1]) != 0 || len(groups[2]) != 2 {
		t.Fatalf("bad groups: %+v", groups)
	}
}
