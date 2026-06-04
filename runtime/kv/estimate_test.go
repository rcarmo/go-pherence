package kv

import "testing"

func TestEstimateTurboQuantKVBytesSkipsLayersAndReportsSavings(t *testing.T) {
	cfg := DefaultTurboQuantConfig()
	cfg.ResidualWindow = 2
	full, estimated := EstimateTurboQuantKVBytes(8, 1, 8, 16, cfg, true, func(i int) bool { return i%2 == 0 })
	if full != 4*16*8*2*4 {
		t.Fatalf("full bytes=%d", full)
	}
	if estimated <= 0 || estimated >= full {
		t.Fatalf("estimated bytes=%d full=%d", estimated, full)
	}
	saved, ratio := TurboQuantKVByteSavings(full, estimated)
	if saved != full-estimated || ratio <= 0 || ratio >= 1 {
		t.Fatalf("saved=%d ratio=%f full=%d estimated=%d", saved, ratio, full, estimated)
	}
	est := EstimateTurboQuantKV(8, 1, 8, 16, cfg, true, func(i int) bool { return i%2 == 0 })
	if est.FullBytes != full || est.EstimatedBytes != estimated || est.KVLayers != 4 || est.ProtectedLayers == 0 {
		t.Fatalf("bad detailed estimate: %+v", est)
	}
	if est.EstimatedTotalBytes != est.EstimatedBytes+est.EstimatedScratchBytes {
		t.Fatalf("bad total estimate: %+v", est)
	}
}

func TestEstimateTurboQuantScratchBytes(t *testing.T) {
	got := EstimateTurboQuantScratchBytes(2, 1, 4, 3, true)
	// Per layer: quant rotated/index (16+4), dequant rotated/index (16+4),
	// sequence K/V scratch (3*4*2*4=96) = 136. Two layers = 272.
	if got != 272 {
		t.Fatalf("scratch bytes=%d want 272", got)
	}
	if got := EstimateTurboQuantScratchBytes(2, 1, 4, 3, false); got != 0 {
		t.Fatalf("disabled scratch bytes=%d want 0", got)
	}
}

func TestEstimateTurboQuantKVBytesDisabledEqualsFull(t *testing.T) {
	full, estimated := EstimateTurboQuantKVBytes(2, 1, 4, 3, DefaultTurboQuantConfig(), false, nil)
	if full != 2*3*4*2*4 || estimated != full {
		t.Fatalf("full=%d estimated=%d", full, estimated)
	}
}
