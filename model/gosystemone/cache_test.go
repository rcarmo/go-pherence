package gosystemone

import "testing"

func TestCacheCanReuseRequiresMatchingPrefixAndCapacity(t *testing.T) {
	shared := []int{1, 2, 3}
	for _, tc := range []struct {
		name               string
		cached             []int
		required, capacity int
		present, want      bool
	}{
		{name: "hit", cached: []int{1, 2, 3}, required: 128, capacity: 256, present: true, want: true},
		{name: "exact_capacity", cached: []int{1, 2, 3}, required: 256, capacity: 256, present: true, want: true},
		{name: "longer_context", cached: []int{1, 2, 3}, required: 257, capacity: 256, present: true},
		{name: "different_prefix", cached: []int{1, 2, 4}, required: 128, capacity: 256, present: true},
		{name: "missing_context", cached: []int{1, 2, 3}, required: 128, capacity: 256},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := cacheCanReuse(shared, tc.cached, tc.required, tc.capacity, tc.present); got != tc.want {
				t.Fatalf("cacheCanReuse=%v want=%v", got, tc.want)
			}
		})
	}
}
