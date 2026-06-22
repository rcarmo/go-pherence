package minicpmv

import "testing"

func TestCurrentSupportSummary(t *testing.T) {
	s := CurrentSupportSummary()
	if s.SupportVersion != SupportVersion || s.RuntimeStatus != RuntimeStatusPending || !s.Capabilities.ConfigParsing || len(s.PendingRuntimeSteps) == 0 {
		t.Fatalf("bad support summary: %+v", s)
	}
	if s.Capabilities.EndToEndGeneration {
		t.Fatalf("support summary should not claim end-to-end generation: %+v", s)
	}
}
