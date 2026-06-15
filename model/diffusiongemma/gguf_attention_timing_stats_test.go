package diffusiongemma

import "testing"

func TestGGUFAttentionTimingStatsSub(t *testing.T) {
	base := ggufAttentionTimingStats{Calls: 1, TotalNS: 1000, ProjNS: 10, NormRoPENS: 20, KVBuildNS: 30, AttnNS: 40, OProjNS: 50}
	now := ggufAttentionTimingStats{Calls: 4, TotalNS: 9000, ProjNS: 110, NormRoPENS: 220, KVBuildNS: 330, AttnNS: 440, OProjNS: 550}
	d := now.Sub(base)
	if d.Calls != 3 || d.TotalNS != 8000 || d.ProjNS != 100 || d.NormRoPENS != 200 || d.KVBuildNS != 300 || d.AttnNS != 400 || d.OProjNS != 500 {
		t.Fatalf("unexpected diff: %+v", d)
	}
}
