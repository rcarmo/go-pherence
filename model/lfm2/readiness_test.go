package lfm2

import "testing"

func TestRuntimeReadinessReportBlockers(t *testing.T) {
	cov := ReferenceCoverage{CompleteRuntimeTrace: true, NumericParityReady: false, PlaceholderValues: []string{"first_token.logit_checksum"}}
	report := NewRuntimeReadinessReport(CurrentRuntimeStatus(), &cov)
	if report.ReadyForExecution || report.RuntimeReady || report.NumericParityReady {
		t.Fatalf("report=%+v", report)
	}
	want := map[string]bool{"cpu_generation_runtime": true, "nvidia_runtime": true, "placeholder:first_token.logit_checksum": true}
	for _, blocker := range report.Blockers {
		delete(want, blocker)
	}
	if len(want) != 0 {
		t.Fatalf("blockers=%v missing=%v", report.Blockers, want)
	}
}

func TestRuntimeReadinessReportReady(t *testing.T) {
	status := RuntimeStatus{CPUGeneration: true, EmbeddingCPU: true, ConvCPU: true, AttentionCPU: true, MoECPU: true, RuntimeImplemented: true}
	cov := ReferenceCoverage{CompleteRuntimeTrace: true, NumericParityReady: true}
	report := NewRuntimeReadinessReport(status, &cov)
	if !report.ReadyForExecution || !report.RuntimeReady || !report.NumericParityReady || len(report.Blockers) != 0 {
		t.Fatalf("report=%+v", report)
	}
}
