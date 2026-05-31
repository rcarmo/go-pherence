package kv

import "testing"

func TestParseCacheTypeBits(t *testing.T) {
	cases := map[string]struct {
		bits    int
		enabled bool
	}{
		"":       {0, false},
		"f16":    {0, false},
		"turbo4": {4, true},
		"turbo2": {2, true},
		"q8_0":   {8, true},
		"Q4_K":   {4, true},
	}
	for name, want := range cases {
		bits, enabled, err := ParseCacheTypeBits(name)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if bits != want.bits || enabled != want.enabled {
			t.Fatalf("%s: bits=%d enabled=%v want %+v", name, bits, enabled, want)
		}
	}
	if _, _, err := ParseCacheTypeBits("turbo9"); err == nil {
		t.Fatal("expected unsupported cache type error")
	}
}

func TestTurboQuantConfigFromCacheTypes(t *testing.T) {
	cfg, enabled, err := TurboQuantConfigFromCacheTypes("turbo4", "turbo2", 256)
	if err != nil {
		t.Fatal(err)
	}
	if !enabled || cfg.KeyBits != 4 || cfg.ValueBits != 2 || cfg.ResidualWindow != 256 {
		t.Fatalf("cfg=%+v enabled=%v", cfg, enabled)
	}
	cfg, enabled, err = TurboQuantConfigFromCacheTypes("f16", "f16", -1)
	if err != nil {
		t.Fatal(err)
	}
	if enabled || cfg.KeyBits != 4 || cfg.ValueBits != 2 || cfg.ResidualWindow != 128 {
		t.Fatalf("full precision cfg=%+v enabled=%v", cfg, enabled)
	}
}
