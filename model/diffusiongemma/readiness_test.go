package diffusiongemma

import (
	"strings"
	"testing"
)

func TestBuildReadinessSummaryCarriesCapabilityReferenceGaps(t *testing.T) {
	caps := Capabilities()
	s := BuildReadinessSummary(caps, &ShardAvailability{Ready: true}, &TensorReadiness{TextReady: true})
	if len(s.Missing) != len(caps.MissingForReference) {
		t.Fatalf("summary missing=%v want capability gaps=%v", s.Missing, caps.MissingForReference)
	}
	for i := range caps.MissingForReference {
		if s.Missing[i] != caps.MissingForReference[i] {
			t.Fatalf("summary missing[%d]=%q want %q", i, s.Missing[i], caps.MissingForReference[i])
		}
	}
	if !strings.Contains(s.String(), "full image-sequence vision reference fixtures") {
		t.Fatalf("summary string %q missing vision reference fixture blocker", s.String())
	}
}

func TestBuildReadinessSummaryAddsInventoryBlockersBeforeReferenceGaps(t *testing.T) {
	caps := Capabilities()
	s := BuildReadinessSummary(caps, nil, &TensorReadiness{TextReady: false})
	wantPrefix := []string{"text tensor readiness", "safetensor shards"}
	for i, want := range wantPrefix {
		if len(s.Missing) <= i || s.Missing[i] != want {
			t.Fatalf("missing prefix=%v want %v full=%v", s.Missing[:min(len(s.Missing), len(wantPrefix))], wantPrefix, s.Missing)
		}
	}
}

func TestReadinessSummaryStringUsesRuntimeLabel(t *testing.T) {
	s := ReadinessSummary{TextScaffoldReady: true, ShardsReady: true, ReferenceComplete: false, RuntimeReady: false, Missing: []string{"broader reference parity fixtures"}}
	got := s.String()
	if !strings.Contains(got, "text_runtime=true") {
		t.Fatalf("summary %q missing text_runtime label", got)
	}
	if strings.Contains(got, "text_scaffold=") {
		t.Fatalf("summary %q still uses text_scaffold label", got)
	}
}
