package mojev

import (
	"reflect"
	"testing"
)

func TestCandidateTreeGroups(t *testing.T) {
	for _, tc := range []struct {
		lengths          []int
		prefix, capacity int
		want             []int
	}{
		{[]int{1, 2, 3}, 4, 10, []int{3}},
		{[]int{1, 2, 3, 4}, 4, 10, []int{3, 4}},
		{[]int{128, 128}, 2, 256, []int{1, 2}},
		{[]int{4, 1, 4, 1, 4}, 2, 8, []int{2, 4, 5}},
	} {
		cs := make([][]int, len(tc.lengths))
		for i, n := range tc.lengths {
			cs[i] = make([]int, n)
		}
		var got []int
		for start := 0; start < len(cs); {
			end := candidateTreeEnd(cs, start, tc.prefix, tc.capacity)
			if end <= start {
				t.Fatal("group does not advance")
			}
			got = append(got, end)
			start = end
		}
		if !reflect.DeepEqual(got, tc.want) {
			t.Fatal(got, tc.want)
		}
		if a := testing.AllocsPerRun(10, func() { _ = candidateTreeEnd(cs, 0, tc.prefix, tc.capacity) }); a != 0 {
			t.Fatal(a)
		}
	}
}
