package qwen3tts

import "testing"

func TestRuntimeReadinessReportBlockers(t *testing.T) {
	cov := ReferenceCoverage{CompleteRuntimeTrace: true, NumericParityReady: false, PlaceholderValues: []string{"talker.logit_checksum"}}
	report := NewRuntimeReadinessReport(CurrentRuntimeStatus(), &cov)
	if report.ReadyForExecution || report.RuntimeReady || report.NumericParityReady {
		t.Fatalf("report=%+v", report)
	}
	want := map[string]bool{"cpu_talker_runtime": true, "cpu_code_predictor_runtime": true, "decoder12hz_runtime": true, "placeholder:talker.logit_checksum": true}
	for _, blocker := range report.Blockers {
		delete(want, blocker)
	}
	if len(want) != 0 {
		t.Fatalf("blockers=%v missing=%v", report.Blockers, want)
	}
}

func TestRuntimeReadinessReportReady(t *testing.T) {
	status := RuntimeStatus{TalkerCPU: true, CodePredictorCPU: true, Decoder12HzCPU: true, RuntimeImplemented: true}
	cov := ReferenceCoverage{CompleteRuntimeTrace: true, NumericParityReady: true}
	report := NewRuntimeReadinessReport(status, &cov)
	if !report.ReadyForExecution || !report.RuntimeReady || !report.NumericParityReady || len(report.Blockers) != 0 {
		t.Fatalf("report=%+v", report)
	}
}
