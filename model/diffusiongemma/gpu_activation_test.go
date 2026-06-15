package diffusiongemma

import "testing"

func TestF32GELUExactMulStatsResetAndSub(t *testing.T) {
	resetF32GELUExactMulStats()
	f32GELUExactMulCounters.calls.Add(2)
	f32GELUExactMulCounters.elements.Add(3)
	f32GELUExactMulCounters.downloadNS.Add(4)
	f32GELUExactMulCounters.geluNS.Add(5)
	f32GELUExactMulCounters.uploadNS.Add(6)
	before := f32GELUExactMulSnapshot()
	if before.Calls != 2 || before.Elements != 3 || before.DownloadNS != 4 || before.GELUNS != 5 || before.UploadNS != 6 {
		t.Fatalf("unexpected exact GELU stats before reset: %+v", before)
	}
	delta := before.Sub(f32GELUExactMulStats{Calls: 1, Elements: 1, DownloadNS: 1, GELUNS: 2, UploadNS: 3})
	if delta.Calls != 1 || delta.Elements != 2 || delta.DownloadNS != 3 || delta.GELUNS != 3 || delta.UploadNS != 3 {
		t.Fatalf("unexpected exact GELU stats delta: %+v", delta)
	}
	resetF32GELUExactMulStats()
	if after := f32GELUExactMulSnapshot(); after != (f32GELUExactMulStats{}) {
		t.Fatalf("exact GELU stats after reset: %+v", after)
	}
}

func TestF32GELUExactMulScratchReuseAndFree(t *testing.T) {
	freeF32GELUExactMulScratch()
	gate, up := ensureF32GELUExactMulScratch(4)
	if len(gate) != 4 || len(up) != 4 {
		t.Fatalf("scratch lens gate=%d up=%d want 4", len(gate), len(up))
	}
	gate[0], up[0] = 1, 2
	gate2, up2 := ensureF32GELUExactMulScratch(2)
	if len(gate2) != 2 || len(up2) != 2 || gate2[0] != 1 || up2[0] != 2 {
		t.Fatalf("scratch was not reused: gate=%v up=%v", gate2, up2)
	}
	freeF32GELUExactMulScratch()
	if f32GELUExactMulScratch.gate != nil || f32GELUExactMulScratch.up != nil {
		t.Fatalf("scratch not freed")
	}
}
