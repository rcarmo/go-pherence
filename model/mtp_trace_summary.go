package model

import (
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
)

func mtpTraceSummaryEnabled() bool {
	v := strings.TrimSpace(strings.ToLower(os.Getenv("GO_PHERENCE_MTP_TRACE_SUMMARY")))
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

func mtpTraceSummaryInt(name string) int {
	v := strings.TrimSpace(os.Getenv(name))
	if v == "" {
		return -1
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return -1
	}
	return n
}

func mtpTraceSummaryRow() int { return mtpTraceSummaryInt("GO_PHERENCE_MTP_TRACE_ROW") }
func mtpTraceSummaryPos() int { return mtpTraceSummaryInt("GO_PHERENCE_MTP_TRACE_POS") }

func traceMTPSummary(label string, row, layer, pos int, x []float32) {
	if !mtpTraceSummaryEnabled() || len(x) == 0 {
		return
	}
	wantRow := mtpTraceSummaryRow()
	if wantRow >= 0 && row >= 0 && row != wantRow {
		return
	}
	wantPos := mtpTraceSummaryPos()
	if wantPos >= 0 && pos != wantPos {
		return
	}
	var sum, sumsq float64
	maxAbs := 0.0
	maxIdx := 0
	for i, v := range x {
		fv := float64(v)
		sum += fv
		sumsq += fv * fv
		av := math.Abs(fv)
		if av > maxAbs {
			maxAbs = av
			maxIdx = i
		}
	}
	first := 4
	if len(x) < first {
		first = len(x)
	}
	fmt.Fprintf(os.Stderr, "GO_MTP_SUMMARY label=%s row=%d layer=%d pos=%d n=%d mean=%.9g rms=%.9g max_abs=%.9g max_idx=%d first=%v\n", label, row, layer, pos, len(x), sum/float64(len(x)), math.Sqrt(sumsq/float64(len(x))), maxAbs, maxIdx, append([]float32(nil), x[:first]...))
}
