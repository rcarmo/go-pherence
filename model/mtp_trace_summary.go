package model

import (
	"encoding/binary"
	"fmt"
	"hash/fnv"
	"math"
	"os"
	"path/filepath"
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
	h := fnv.New64a()
	var buf [4]byte
	for i, v := range x {
		binary.LittleEndian.PutUint32(buf[:], math.Float32bits(v))
		_, _ = h.Write(buf[:])
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
	fmt.Fprintf(os.Stderr, "GO_MTP_SUMMARY label=%s row=%d layer=%d pos=%d n=%d mean=%.9g rms=%.9g max_abs=%.9g max_idx=%d hash=%016x first=%v\n", label, row, layer, pos, len(x), sum/float64(len(x)), math.Sqrt(sumsq/float64(len(x))), maxAbs, maxIdx, h.Sum64(), append([]float32(nil), x[:first]...))
	traceMTPSummaryDump(label, row, layer, pos, x)
}

func traceMTPSummaryDump(label string, row, layer, pos int, x []float32) {
	dir := strings.TrimSpace(os.Getenv("GO_PHERENCE_MTP_TRACE_DUMP_DIR"))
	if dir == "" || len(x) == 0 {
		return
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		fmt.Fprintf(os.Stderr, "GO_MTP_SUMMARY_DUMP_ERROR mkdir=%q err=%v\n", dir, err)
		return
	}
	safeLabel := strings.NewReplacer("/", "_", "\\", "_", " ", "_", "(", "_", ")", "_", ":", "_").Replace(label)
	path := filepath.Join(dir, fmt.Sprintf("go_%s_layer%02d_pos%d_row%d_n%d.f32", safeLabel, layer, pos, row, len(x)))
	f, err := os.Create(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "GO_MTP_SUMMARY_DUMP_ERROR create=%q err=%v\n", path, err)
		return
	}
	defer f.Close()
	var buf [4]byte
	for _, v := range x {
		binary.LittleEndian.PutUint32(buf[:], math.Float32bits(v))
		if _, err := f.Write(buf[:]); err != nil {
			fmt.Fprintf(os.Stderr, "GO_MTP_SUMMARY_DUMP_ERROR write=%q err=%v\n", path, err)
			return
		}
	}
	fmt.Fprintf(os.Stderr, "GO_MTP_SUMMARY_DUMP path=%s\n", path)
}
