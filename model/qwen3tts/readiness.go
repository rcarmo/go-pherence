package qwen3tts

import modelreadiness "github.com/rcarmo/go-pherence/model/internal/readiness"

// RuntimeReadinessReport combines runtime implementation status with fixture
// parity coverage. It is a validation/reporting surface only; it does not run
// generation.
type RuntimeReadinessReport struct {
	RuntimeStatus      RuntimeStatus      `json:"runtime_status"`
	ReferenceCoverage  *ReferenceCoverage `json:"reference_coverage,omitempty"`
	RuntimeReady       bool               `json:"runtime_ready"`
	NumericParityReady bool               `json:"numeric_parity_ready"`
	ReadyForExecution  bool               `json:"ready_for_execution"`
	Blockers           []string           `json:"blockers,omitempty"`
}

func NewRuntimeReadinessReport(status RuntimeStatus, coverage *ReferenceCoverage) RuntimeReadinessReport {
	var numericParityReady bool
	var missing, placeholders []string
	if coverage != nil {
		numericParityReady = coverage.NumericParityReady
		missing = coverage.Missing
		placeholders = coverage.PlaceholderValues
	}
	core := modelreadiness.BuildReportCore(status.RuntimeImplemented, status.Pending, coverage != nil, numericParityReady, missing, placeholders)
	return RuntimeReadinessReport{RuntimeStatus: status, ReferenceCoverage: coverage, RuntimeReady: core.RuntimeReady, NumericParityReady: core.NumericParityReady, ReadyForExecution: core.ReadyForExecution, Blockers: core.Blockers}
}
