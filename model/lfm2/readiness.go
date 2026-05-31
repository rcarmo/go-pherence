package lfm2

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
	report := RuntimeReadinessReport{RuntimeStatus: status, ReferenceCoverage: coverage, RuntimeReady: status.RuntimeImplemented}
	if coverage != nil {
		report.NumericParityReady = coverage.NumericParityReady
	}
	if !report.RuntimeReady {
		report.Blockers = append(report.Blockers, status.Pending...)
	}
	if coverage == nil {
		report.Blockers = append(report.Blockers, "reference_fixture_missing")
	} else {
		for _, missing := range coverage.Missing {
			report.Blockers = append(report.Blockers, "reference_missing:"+missing)
		}
		for _, placeholder := range coverage.PlaceholderValues {
			report.Blockers = append(report.Blockers, "placeholder:"+placeholder)
		}
	}
	report.ReadyForExecution = report.RuntimeReady && report.NumericParityReady && len(report.Blockers) == 0
	return report
}
