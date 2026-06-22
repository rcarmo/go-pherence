package minicpmv

import "testing"

func TestValidateSupportSummary(t *testing.T) {
	if err := ValidateSupportSummary(CurrentSupportSummary()); err != nil {
		t.Fatalf("ValidateSupportSummary: %v", err)
	}
}

func TestValidateSupportSummaryRejectsRuntimeClaim(t *testing.T) {
	s := CurrentSupportSummary()
	s.Capabilities.EndToEndGeneration = true
	if err := ValidateSupportSummary(s); err == nil {
		t.Fatalf("expected runtime claim to fail")
	}
}

func TestValidateSupportSummaryRejectsMissingPendingSteps(t *testing.T) {
	s := CurrentSupportSummary()
	s.PendingRuntimeSteps = nil
	if err := ValidateSupportSummary(s); err == nil {
		t.Fatalf("expected missing pending steps to fail")
	}
}
