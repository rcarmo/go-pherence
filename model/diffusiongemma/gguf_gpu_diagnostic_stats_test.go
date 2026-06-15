package diffusiongemma

import "testing"

func TestResetGGUFGPUDiagnosticStats(t *testing.T) {
	ResetGGUFGPUDiagnosticStats()
	ggufExpertDispatchCounters.fusedUsed.Add(1)
	ggufExpertDispatchCounters.q4BudgetBytes.Add(2)
	ggufExpertDispatchCounters.gpuAttemptNS.Add(3)
	ggufChunkedLMHeadCounters.chunks.Add(4)
	ggufChunkedLMHeadCounters.uploadNS.Add(5)
	ggufTempDenseUploadCounters.calls.Add(6)
	ggufTempDenseUploadCounters.encoderMLPHits.Add(7)
	ggufAttentionTimingCounters.calls.Add(8)
	ggufAttentionTimingCounters.totalNS.Add(9)

	if s := ggufExpertDispatchStatsSnapshot(); s.FusedUsed != 1 || s.Q4BudgetBytes != 2 || s.GPUAttemptNS != 3 {
		t.Fatalf("expert stats before reset=%+v", s)
	}
	if s := ggufChunkedLMHeadSnapshot(); s.Chunks != 4 || s.UploadNS != 5 {
		t.Fatalf("lmhead stats before reset=%+v", s)
	}
	if s := ggufTempDenseUploadSnapshot(); s.Calls != 6 || s.EncoderMLPHits != 7 {
		t.Fatalf("temp dense stats before reset=%+v", s)
	}
	if s := ggufAttentionTimingSnapshot(); s.Calls != 8 || s.TotalNS != 9 {
		t.Fatalf("attention stats before reset=%+v", s)
	}

	ResetGGUFGPUDiagnosticStats()
	if s := ggufExpertDispatchStatsSnapshot(); s != (ggufExpertDispatchStats{}) {
		t.Fatalf("expert stats after reset=%+v", s)
	}
	if s := ggufChunkedLMHeadSnapshot(); s != (ggufChunkedLMHeadStats{}) {
		t.Fatalf("lmhead stats after reset=%+v", s)
	}
	if s := ggufTempDenseUploadSnapshot(); s != (ggufTempDenseUploadStats{}) {
		t.Fatalf("temp dense stats after reset=%+v", s)
	}
	if s := ggufAttentionTimingSnapshot(); s != (ggufAttentionTimingStats{}) {
		t.Fatalf("attention stats after reset=%+v", s)
	}
}
