package diffusiongemma

import "testing"

func TestSummarizeSelectedExpertGroupedWork(t *testing.T) {
	items := []SelectedExpertWorkItem{{Position: 0, Expert: 2}, {Position: 1, Expert: 2}, {Position: 0, Expert: 5}, {Position: 2, Expert: 5}, {Position: 3, Expert: 5}, {Position: 1, Expert: 1}}
	arr := BuildSelectedExpertWorkArrays(items)
	grouped, err := BuildSelectedExpertGroupedWork(arr, 8)
	if err != nil {
		t.Fatal(err)
	}
	s, err := SummarizeSelectedExpertGroupedWork(grouped, arr.Len())
	if err != nil {
		t.Fatal(err)
	}
	if s.WorkItems != 6 || s.ActiveExperts != 3 || s.MaxGroup != 3 || s.MinGroup != 1 {
		t.Fatalf("summary=%+v", s)
	}
}
