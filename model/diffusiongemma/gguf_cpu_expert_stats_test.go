package diffusiongemma

import "testing"

func TestResetGGUFCPUExpertTimingStats(t *testing.T) {
	ResetGGUFCPUExpertTimingStats()
	ggufCPUExpertTimingCounters.calls.Add(2)
	ggufCPUExpertTimingCounters.positions.Add(3)
	ggufCPUExpertTimingCounters.workItems.Add(4)
	ggufCPUExpertTimingCounters.activeExperts.Add(5)
	ggufCPUExpertTimingCounters.gateNS.Add(6)
	before := ggufCPUExpertTimingSnapshot()
	if before.Calls != 2 || before.Positions != 3 || before.WorkItems != 4 || before.ActiveExperts != 5 || before.GateNS != 6 {
		t.Fatalf("unexpected stats before reset: %+v", before)
	}
	ResetGGUFCPUExpertTimingStats()
	after := ggufCPUExpertTimingSnapshot()
	if after != (ggufCPUExpertTimingStats{}) {
		t.Fatalf("stats after reset=%+v, want zero", after)
	}
}
