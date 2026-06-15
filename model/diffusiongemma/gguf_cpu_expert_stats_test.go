package diffusiongemma

import "testing"

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
	cases := map[int]int{0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 8: 2, 9: 3, 12: 3, 13: 4, 15: 4, 16: 5, 64: 5}
	for nPos, want := range cases {
		if got := ggufCPUExpertBatchBucket(nPos); got != want {
			t.Fatalf("bucket(%d)=%d, want %d", nPos, got, want)
		}
	}
	if got := ggufCPUExpertBatchBucketsString(ggufCPUExpertBatchBuckets{1, 2, 3, 4, 5, 6}); got != "1:1,2-3:2,4-8:3,9-12:4,13-15:5,16+:6" {
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
	ggufCPUExpertTimingCounters.q4DirectBatches[0].Add(10)
	ggufCPUExpertTimingCounters.q4DequantBatches[1].Add(11)
	ggufCPUExpertTimingCounters.q8DirectBatches[2].Add(12)
	ggufCPUExpertTimingCounters.q8DequantBatches[5].Add(13)
	ggufCPUExpertTimingCounters.gateNS.Add(14)
	before := ggufCPUExpertTimingSnapshot()
	if before.Calls != 2 || before.Positions != 3 || before.WorkItems != 4 || before.ActiveExperts != 5 || before.Q4DirectRows != 6 || before.Q4DequantRows != 7 || before.Q8DirectRows != 8 || before.Q8DequantRows != 9 || before.Q4DirectBatches[0] != 10 || before.Q4DequantBatches[1] != 11 || before.Q8DirectBatches[2] != 12 || before.Q8DequantBatches[5] != 13 || before.GateNS != 14 {
		t.Fatalf("unexpected stats before reset: %+v", before)
	}
	base := before
	ggufCPUExpertTimingCounters.q4DirectBatches[0].Add(3)
	ggufCPUExpertTimingCounters.q8DequantBatches[5].Add(5)
	delta := ggufCPUExpertTimingSnapshot().Sub(base)
	if delta.Q4DirectBatches[0] != 3 || delta.Q8DequantBatches[5] != 5 {
		t.Fatalf("unexpected stats delta: %+v", delta)
	}
	ResetGGUFCPUExpertTimingStats()
	after := ggufCPUExpertTimingSnapshot()
	if after != (ggufCPUExpertTimingStats{}) {
		t.Fatalf("stats after reset=%+v, want zero", after)
	}
}
