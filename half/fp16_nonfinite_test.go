package half

import (
	"math"
	"testing"
)

func TestF32ToF16NaNClassificationSignAndQuieting(t *testing.T) {
	for u := 0; u <= 0xffff; u++ {
		h := uint16(u)
		if h&0x7c00 != 0x7c00 || h&0x03ff == 0 {
			continue
		}
		got := F32ToF16(F16ToF32(h))
		want := h | 0x0200
		if got != want || !math.IsNaN(float64(F16ToF32(got))) {
			t.Fatalf("half NaN %04x -> %04x want quieted %04x", h, got, want)
		}
	}
	for _, bits := range []uint32{0x7f800001, 0x7f801fff, 0x7fffffff, 0xff800001, 0xff801fff, 0xffffffff} {
		got := F32ToF16(math.Float32frombits(bits))
		if got&0x7c00 != 0x7c00 || got&0x0200 == 0 || got&0x8000 != uint16(bits>>16)&0x8000 {
			t.Fatalf("float32 NaN %08x -> %04x", bits, got)
		}
	}
}

func TestF32ToF16PreservesLegacyFiniteRounding(t *testing.T) {
	// Finite ties retain the existing nearest/ties-away policy, not BF16's RNE.
	for _, c := range []struct {
		bits uint32
		want uint16
	}{
		{0, 0}, {0x80000000, 0x8000}, {0x7f800000, 0x7c00}, {0xff800000, 0xfc00},
		{0x3f801000, 0x3c01}, {0xbf801000, 0xbc01}, // half-way above +/-1
		{0x33000000, 0x0001}, {0xb3000000, 0x8001}, // half smallest subnormal
		{0x7f7fffff, 0x7c00}, {0xff7fffff, 0xfc00},
	} {
		if got := F32ToF16(math.Float32frombits(c.bits)); got != c.want {
			t.Fatalf("%08x -> %04x want %04x", c.bits, got, c.want)
		}
	}
}
