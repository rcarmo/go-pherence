package diffusiongemma

import "testing"

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
	ggufCPUExpertTimingCounters.gateNS.Add(10)
	before := ggufCPUExpertTimingSnapshot()
	if before.Calls != 2 || before.Positions != 3 || before.WorkItems != 4 || before.ActiveExperts != 5 || before.Q4DirectRows != 6 || before.Q4DequantRows != 7 || before.Q8DirectRows != 8 || before.Q8DequantRows != 9 || before.GateNS != 10 {
		t.Fatalf("unexpected stats before reset: %+v", before)
	}
	ResetGGUFCPUExpertTimingStats()
	after := ggufCPUExpertTimingSnapshot()
	if after != (ggufCPUExpertTimingStats{}) {
		t.Fatalf("stats after reset=%+v, want zero", after)
	}
}
