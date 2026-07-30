package model

import "testing"

func TestPrefillChunkPlanStructSpans(t *testing.T) {
	plan := PrefillChunkPlan{Spans: []PrefillChunkSpan{{Start: 0, End: 32}, {Start: 32, End: 65}}}
	got := plan.StructSpans()
	if len(got) != len(plan.Spans) {
		t.Fatalf("len=%d want %d", len(got), len(plan.Spans))
	}
	for i, span := range plan.Spans {
		if got[i].Start != span.Start || got[i].End != span.End {
			t.Fatalf("span %d=%+v want %+v", i, got[i], span)
		}
	}
	got[0].Start = 9
	if plan.Spans[0].Start != 0 {
		t.Fatalf("adapter aliased plan spans: %+v", plan.Spans)
	}
}
