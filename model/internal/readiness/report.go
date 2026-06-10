package readiness

// ReportCore contains the model-agnostic readiness booleans and blockers used
// by model-specific public report structs.
type ReportCore struct {
	RuntimeReady       bool
	NumericParityReady bool
	ReadyForExecution  bool
	Blockers           []string
}

func BuildReportCore(runtimeImplemented bool, pending []string, coveragePresent bool, numericParityReady bool, missing []string, placeholders []string) ReportCore {
	report := ReportCore{RuntimeReady: runtimeImplemented, NumericParityReady: numericParityReady}
	if !report.RuntimeReady {
		report.Blockers = append(report.Blockers, pending...)
	}
	if !coveragePresent {
		report.Blockers = append(report.Blockers, "reference_fixture_missing")
	} else {
		for _, item := range missing {
			report.Blockers = append(report.Blockers, "reference_missing:"+item)
		}
		for _, item := range placeholders {
			report.Blockers = append(report.Blockers, "placeholder:"+item)
		}
	}
	report.ReadyForExecution = report.RuntimeReady && report.NumericParityReady && len(report.Blockers) == 0
	return report
}
