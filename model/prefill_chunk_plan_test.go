package model

import (
	"strings"
	"testing"
)

func TestNewPrefillChunkPlanSelectsLargestAllowedChunk(t *testing.T) {
	dims := PrefillChunkModelDims{HiddenSize: 1, QDim: 1, KVDim: 1, Intermediate: 1, Layers: 1}
	estimate, err := EstimatePrefillChunkScratch(dims)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name      string
		budgetFor int
		wantChunk int
		wantSpans []PrefillChunkSpan
	}{
		{
			name:      "32",
			budgetFor: 32,
			wantChunk: 32,
			wantSpans: []PrefillChunkSpan{{0, 32}, {32, 64}, {64, 96}, {96, 128}, {128, 130}},
		},
		{
			name:      "64",
			budgetFor: 64,
			wantChunk: 64,
			wantSpans: []PrefillChunkSpan{{0, 64}, {64, 128}, {128, 130}},
		},
		{
			name:      "128",
			budgetFor: 128,
			wantChunk: 128,
			wantSpans: []PrefillChunkSpan{{0, 128}, {128, 130}},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			budget, err := estimate.TotalBytes(tc.budgetFor)
			if err != nil {
				t.Fatal(err)
			}
			plan, err := NewPrefillChunkPlan(130, dims, budget, nil)
			if err != nil {
				t.Fatal(err)
			}
			if plan.ChunkSize != tc.wantChunk {
				t.Fatalf("chunk size=%d, want %d", plan.ChunkSize, tc.wantChunk)
			}
			if plan.PeakRows != tc.wantChunk {
				t.Fatalf("peak rows=%d, want %d", plan.PeakRows, tc.wantChunk)
			}
			if plan.PeakScratchBytes != budget {
				t.Fatalf("peak scratch=%d, want %d", plan.PeakScratchBytes, budget)
			}
			if !samePrefillChunkSpans(plan.Spans, tc.wantSpans) {
				t.Fatalf("spans=%v, want %v", plan.Spans, tc.wantSpans)
			}
		})
	}
}

func TestNewPrefillChunkPlanPreservesTailAbsolutePositions(t *testing.T) {
	dims := PrefillChunkModelDims{HiddenSize: 1, QDim: 1, KVDim: 1, Intermediate: 1, Layers: 1}
	estimate, err := EstimatePrefillChunkScratch(dims)
	if err != nil {
		t.Fatal(err)
	}
	budget, err := estimate.TotalBytes(64)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := NewPrefillChunkPlan(70, dims, budget, []int{32, 64, 128})
	if err != nil {
		t.Fatal(err)
	}
	want := []PrefillChunkSpan{{0, 64}, {64, 70}}
	if !samePrefillChunkSpans(plan.Spans, want) {
		t.Fatalf("spans=%v, want %v", plan.Spans, want)
	}
}

func TestNewPrefillChunkPlanZeroTokens(t *testing.T) {
	dims := PrefillChunkModelDims{HiddenSize: 1, QDim: 1, KVDim: 1, Intermediate: 1, Layers: 1}
	plan, err := NewPrefillChunkPlan(0, dims, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	if plan.ChunkSize != 0 {
		t.Fatalf("chunk size=%d, want 0", plan.ChunkSize)
	}
	if plan.PeakRows != 0 || plan.PeakScratchBytes != 0 {
		t.Fatalf("peak rows/scratch=%d/%d, want 0/0", plan.PeakRows, plan.PeakScratchBytes)
	}
	if len(plan.Spans) != 0 {
		t.Fatalf("spans=%v, want empty", plan.Spans)
	}
}

func TestNewPrefillChunkPlanUnlimitedDisablesChunking(t *testing.T) {
	dims := PrefillChunkModelDims{HiddenSize: 1, QDim: 1, KVDim: 1, Intermediate: 1, Layers: 1}
	estimate, err := EstimatePrefillChunkScratch(dims)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := NewPrefillChunkPlan(70, dims, PrefillChunkBudgetUnlimited, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !plan.Disabled {
		t.Fatal("plan.Disabled=false, want true")
	}
	if plan.ChunkSize != 70 {
		t.Fatalf("chunk size=%d, want 70", plan.ChunkSize)
	}
	wantPeak, err := estimate.TotalBytes(70)
	if err != nil {
		t.Fatal(err)
	}
	if plan.PeakScratchBytes != wantPeak {
		t.Fatalf("peak scratch=%d, want %d", plan.PeakScratchBytes, wantPeak)
	}
	want := []PrefillChunkSpan{{0, 70}}
	if !samePrefillChunkSpans(plan.Spans, want) {
		t.Fatalf("spans=%v, want %v", plan.Spans, want)
	}
}

func TestNewPrefillChunkPlanOverflow(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	dims := PrefillChunkModelDims{HiddenSize: maxInt/6 + 1, QDim: 1, KVDim: 1, Intermediate: 1, Layers: 1}
	_, err := NewPrefillChunkPlan(1, dims, PrefillChunkBudgetUnlimited, nil)
	if err == nil || !strings.Contains(err.Error(), "overflow") {
		t.Fatalf("err=%v, want overflow", err)
	}
}

func BenchmarkNewPrefillChunkPlan(b *testing.B) {
	dims := PrefillChunkModelDims{HiddenSize: 4096, QDim: 4096, KVDim: 1024, Intermediate: 14336, Layers: 32}
	estimate, err := EstimatePrefillChunkScratch(dims)
	if err != nil {
		b.Fatal(err)
	}
	budget, err := estimate.TotalBytes(64)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	var sink PrefillChunkPlan
	for i := 0; i < b.N; i++ {
		plan, err := NewPrefillChunkPlan(1537, dims, budget, nil)
		if err != nil {
			b.Fatal(err)
		}
		if plan.ChunkSize != 64 {
			b.Fatalf("chunk size=%d, want 64", plan.ChunkSize)
		}
		sink = plan
	}
	_ = sink
}

func samePrefillChunkSpans(a, b []PrefillChunkSpan) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
