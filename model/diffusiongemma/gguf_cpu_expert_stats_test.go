package diffusiongemma

import (
	"math"
	"testing"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

func TestGGUFExpertSdotBatchToMatchesSdot(t *testing.T) {
	for _, nPos := range []int{1, 2, 3, 4, 8, 9, 16} {
		inDim := 64
		dstStride := 11
		w := make([]float32, inDim)
		x := make([]float32, nPos*inDim)
		dst := make([]float32, nPos*dstStride)
		for i := range w {
			w[i] = float32((i%13)-6) * 0.03125
		}
		for i := range x {
			x[i] = float32((i%17)-8) * 0.015625
		}
		if !ggufExpertSdotBatchTo(w, x, nPos, inDim, dst, dstStride) {
			t.Fatalf("Sdot batch rejected nPos=%d", nPos)
		}
		for pos := 0; pos < nPos; pos++ {
			want := simd.Sdot(w, x[pos*inDim:(pos+1)*inDim])
			got := dst[pos*dstStride]
			if math.Abs(float64(got-want)) > 1e-4 {
				t.Fatalf("nPos=%d pos=%d got=%g want=%g", nPos, pos, got, want)
			}
		}
	}
}

func TestGGUFCPUExpertLayerTraceCanBeEnabled(t *testing.T) {
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_GGUF_CPU_EXPERT_LAYER_TRACE", "")
	if diffusionGemmaGGUFCPUExpertLayerTraceEnabled() {
		t.Fatal("CPU expert layer trace should default off")
	}
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_GGUF_CPU_EXPERT_LAYER_TRACE", "1")
	if !diffusionGemmaGGUFCPUExpertLayerTraceEnabled() {
		t.Fatal("CPU expert layer trace did not honor enable")
	}
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_GGUF_CPU_EXPERT_LAYER_TRACE", "false")
	if diffusionGemmaGGUFCPUExpertLayerTraceEnabled() {
		t.Fatal("CPU expert layer trace did not honor disable")
	}
}

func TestGGUFCPUDirectQuantPolicyDefaultsOnAndCanBeDisabled(t *testing.T) {
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_GGUF_CPU_Q4_DIRECT", "")
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_GGUF_CPU_Q8_DIRECT", "")
	if !diffusionGemmaGGUFCPUQ4DirectEnabled() || !diffusionGemmaGGUFCPUQ8DirectEnabled() {
		t.Fatal("direct quant policy should default on; platform/nPos gates decide actual use")
	}
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_GGUF_CPU_Q4_DIRECT", "0")
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_GGUF_CPU_Q8_DIRECT", "false")
	if diffusionGemmaGGUFCPUQ4DirectEnabled() || diffusionGemmaGGUFCPUQ8DirectEnabled() {
		t.Fatal("direct quant policy did not honor explicit disable")
	}
}

func TestGGUFCPUExpertBatchBuckets(t *testing.T) {
	cases := map[int]int{0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 12: 7, 13: 8, 15: 8, 16: 9, 64: 9}
	for nPos, want := range cases {
		if got := ggufCPUExpertBatchBucket(nPos); got != want {
			t.Fatalf("bucket(%d)=%d, want %d", nPos, got, want)
		}
	}
	if got := ggufCPUExpertBatchBucketsString(ggufCPUExpertBatchBuckets{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}); got != "1:1,2-3:2,4:3,5:4,6:5,7:6,8:7,9-12:8,13-15:9,16+:10" {
		t.Fatalf("unexpected bucket string %q", got)
	}
}

func TestResetGGUFCPUExpertTimingStats(t *testing.T) {
	ResetGGUFCPUExpertTimingStats()
	ggufCPUExpertTimingCounters.calls.Add(2)
	ggufCPUExpertTimingCounters.positions.Add(3)
	ggufCPUExpertTimingCounters.workItems.Add(4)
	ggufCPUExpertTimingCounters.activeExperts.Add(5)
	ggufCPUExpertTimingCounters.q4DirectRows.Add(6)
	ggufCPUExpertTimingCounters.q4DequantRows.Add(7)
	ggufCPUExpertTimingCounters.q8DirectRows.Add(8)
	ggufCPUExpertTimingCounters.q8DequantRows.Add(9)
	ggufCPUExpertTimingCounters.q5DequantRows.Add(10)
	ggufCPUExpertTimingCounters.q4DirectBatches[0].Add(11)
	ggufCPUExpertTimingCounters.q4DequantBatches[1].Add(12)
	ggufCPUExpertTimingCounters.q8DirectBatches[2].Add(13)
	ggufCPUExpertTimingCounters.q8DequantBatches[9].Add(14)
	ggufCPUExpertTimingCounters.q5DequantBatches[3].Add(15)
	ggufCPUExpertTimingCounters.gateNS.Add(16)
	before := ggufCPUExpertTimingSnapshot()
	if before.Calls != 2 || before.Positions != 3 || before.WorkItems != 4 || before.ActiveExperts != 5 || before.Q4DirectRows != 6 || before.Q4DequantRows != 7 || before.Q8DirectRows != 8 || before.Q8DequantRows != 9 || before.Q5DequantRows != 10 || before.Q4DirectBatches[0] != 11 || before.Q4DequantBatches[1] != 12 || before.Q8DirectBatches[2] != 13 || before.Q8DequantBatches[9] != 14 || before.Q5DequantBatches[3] != 15 || before.GateNS != 16 {
		t.Fatalf("unexpected stats before reset: %+v", before)
	}
	base := before
	ggufCPUExpertTimingCounters.q4DirectBatches[0].Add(3)
	ggufCPUExpertTimingCounters.q8DequantBatches[9].Add(5)
	ggufCPUExpertTimingCounters.q5DequantBatches[3].Add(7)
	delta := ggufCPUExpertTimingSnapshot().Sub(base)
	if delta.Q4DirectBatches[0] != 3 || delta.Q8DequantBatches[9] != 5 || delta.Q5DequantBatches[3] != 7 {
		t.Fatalf("unexpected stats delta: %+v", delta)
	}
	ResetGGUFCPUExpertTimingStats()
	after := ggufCPUExpertTimingSnapshot()
	if after != (ggufCPUExpertTimingStats{}) {
		t.Fatalf("stats after reset=%+v, want zero", after)
	}
}
