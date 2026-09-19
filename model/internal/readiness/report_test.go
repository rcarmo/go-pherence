package readiness

import "testing"

func TestReadinessRequiresRuntimeParityAndCoverage(t *testing.T) {
	for _, runtime := range []bool{false, true} {
		for _, parity := range []bool{false, true} {
			for _, coverage := range []bool{false, true} {
				got := BuildReportCore(runtime, []string{"kernel"}, coverage, parity, nil, nil)
				if got.ReadyForExecution != (runtime && parity && coverage) {
					t.Fatal(got)
				}
			}
		}
	}
	for _, missing := range [][]string{{"weight"}, nil} {
		for _, placeholder := range [][]string{{"pending"}, nil} {
			r := BuildReportCore(true, nil, true, true, missing, placeholder)
			if r.ReadyForExecution != (len(missing)+len(placeholder) == 0) {
				t.Fatal(r)
			}
		}
	}
}
