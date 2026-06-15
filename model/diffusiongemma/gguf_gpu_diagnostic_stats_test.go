package diffusiongemma

import "testing"

func TestResetGGUFGPUDiagnosticStats(t *testing.T) {
	ResetGGUFGPUDiagnosticStats()
	ggufExpertDispatchCounters.fusedUsed.Add(1)
	ggufExpertDispatchCounters.q4RawPointerTable.Add(11)
	ggufExpertDispatchCounters.q4BudgetBytes.Add(2)
	ggufExpertDispatchCounters.activeSetCalls.Add(2)
	ggufExpertDispatchCounters.activeSetExperts.Add(10)
	ggufExpertDispatchCounters.activeSetMaxExperts.Store(7)
	ggufExpertDispatchCounters.activeSetWorkItems.Add(24)
	ggufExpertDispatchCounters.activeSetMaxWorkItems.Store(16)
	ggufExpertDispatchCounters.q4MissingExperts.Add(6)
	ggufExpertDispatchCounters.q4MissingMaxExperts.Store(4)
	ggufExpertDispatchCounters.q4MissingBytes.Add(8 * 1024 * 1024)
	ggufExpertDispatchCounters.q4MissingMaxBytes.Store(5 * 1024 * 1024)
	ggufExpertDispatchCounters.q4MissingBudgetExceeds.Add(1)
	ggufExpertDispatchCounters.gpuAttemptNS.Add(3)
	ggufChunkedLMHeadCounters.chunks.Add(4)
	ggufChunkedLMHeadCounters.uploadNS.Add(5)
	ggufTempDenseUploadCounters.calls.Add(6)
	ggufTempDenseUploadCounters.encoderMLPHits.Add(7)
	ggufAttentionTimingCounters.calls.Add(8)
	ggufAttentionTimingCounters.totalNS.Add(9)

	if s := ggufExpertDispatchStatsSnapshot(); s.FusedUsed != 1 || s.Q4RawPointerTable != 11 || s.Q4BudgetBytes != 2 || s.GPUAttemptNS != 3 || s.ActiveSetCalls != 2 || s.ActiveSetExperts != 10 || s.ActiveSetMaxExperts != 7 || s.ActiveSetWorkItems != 24 || s.ActiveSetMaxWorkItems != 16 || s.Q4MissingExperts != 6 || s.Q4MissingMaxExperts != 4 || s.Q4MissingBytes != 8*1024*1024 || s.Q4MissingMaxBytes != 5*1024*1024 || s.Q4MissingBudgetExceeds != 1 {
		t.Fatalf("expert stats before reset=%+v", s)
	} else if activeAvg, workAvg, missingAvg, missingMiB, missingMaxMiB := s.ActiveSetSummary(); activeAvg != 5 || workAvg != 12 || missingAvg != 3 || missingMiB != 8 || missingMaxMiB != 5 {
		t.Fatalf("expert summary active=%g work=%g missing=%g mib=%g max=%g", activeAvg, workAvg, missingAvg, missingMiB, missingMaxMiB)
	} else if delta := s.Sub(ggufExpertDispatchStats{ActiveSetCalls: 1, ActiveSetExperts: 4, ActiveSetMaxExperts: 7, ActiveSetWorkItems: 8, ActiveSetMaxWorkItems: 16, Q4MissingExperts: 2, Q4MissingMaxExperts: 4, Q4MissingBytes: 2 * 1024 * 1024, Q4MissingMaxBytes: 5 * 1024 * 1024, Q4MissingBudgetExceeds: 1}); delta.ActiveSetCalls != 1 || delta.ActiveSetExperts != 6 || delta.ActiveSetMaxExperts != 0 || delta.ActiveSetWorkItems != 16 || delta.ActiveSetMaxWorkItems != 0 || delta.Q4MissingExperts != 4 || delta.Q4MissingMaxExperts != 0 || delta.Q4MissingBytes != 6*1024*1024 || delta.Q4MissingMaxBytes != 0 || delta.Q4MissingBudgetExceeds != 0 {
		t.Fatalf("expert stats delta=%+v", delta)
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
