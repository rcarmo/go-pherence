package minicpmv

import "testing"

func TestPendingRuntimeSteps(t *testing.T) {
	steps := PendingRuntimeSteps()
	if len(steps) < 5 {
		t.Fatalf("expected runtime steps, got %v", steps)
	}
	for i, step := range steps {
		if step == "" {
			t.Fatalf("empty step at %d", i)
		}
	}
}
