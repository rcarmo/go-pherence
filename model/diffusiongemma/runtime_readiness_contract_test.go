package diffusiongemma

import "testing"

func TestRuntimeReadinessContractRequiresReferenceGaps(t *testing.T) {
	caps := Capabilities()
	if !caps.TextOnlyScaffoldReady {
		t.Fatalf("text runtime should remain implemented/ready while global readiness is gated: %+v", caps)
	}
	if caps.ImplementedOps != caps.TotalOps {
		t.Fatalf("all semantic ops should be implemented before readiness is only reference-gated: implemented=%d total=%d", caps.ImplementedOps, caps.TotalOps)
	}
	if caps.ReferenceComplete || caps.RuntimeReady {
		t.Fatalf("global readiness must remain false until reference gaps close: %+v", caps)
	}
	gaps := MissingReferenceGaps()
	if len(caps.MissingForReference) != len(gaps) {
		t.Fatalf("missing reference gaps=%v want %v", caps.MissingForReference, gaps)
	}
	for i := range gaps {
		if caps.MissingForReference[i] != gaps[i] {
			t.Fatalf("missing gap[%d]=%q want %q", i, caps.MissingForReference[i], gaps[i])
		}
	}
}
