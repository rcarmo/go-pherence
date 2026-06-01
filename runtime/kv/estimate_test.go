package kv

import "testing"

func TestEstimateTurboQuantKVBytesSkipsLayersAndReportsSavings(t *testing.T) {
	cfg := TurboQuantConfig{KeyBits: 4, ValueBits: 2, ResidualWindow: 2}
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
}

func TestEstimateTurboQuantKVBytesDisabledEqualsFull(t *testing.T) {
	full, estimated := EstimateTurboQuantKVBytes(2, 1, 4, 3, DefaultTurboQuantConfig(), false, nil)
	if full != 2*3*4*2*4 || estimated != full {
		t.Fatalf("full=%d estimated=%d", full, estimated)
	}
}
