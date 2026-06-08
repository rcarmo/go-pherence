package half

import (
	"math"
	"testing"
)

func TestF16ToF32Known(t *testing.T) {
	cases := []struct {
		bits uint16
		want float32
	}{
		{0x0000, 0},
		{0x3C00, 1},
		{0xBC00, -1},
		{0x4000, 2},
		{0x7C00, float32(math.Inf(1))},
		{0xFC00, float32(math.Inf(-1))},
		{0x0001, float32(math.Ldexp(1, -24))}, // smallest subnormal
	}
	for _, c := range cases {
		got := F16ToF32(c.bits)
		if math.Float32bits(got) != math.Float32bits(c.want) {
			t.Errorf("F16ToF32(0x%04x) = %v, want %v", c.bits, got, c.want)
		}
	}
	if !math.IsNaN(float64(F16ToF32(0x7E00))) {
		t.Errorf("F16ToF32(0x7E00) should be NaN")
	}
}

func TestF32ToF16Known(t *testing.T) {
	cases := []struct {
		f    float32
		want uint16
	}{
		{0, 0x0000},
		{1, 0x3C00},
		{-1, 0xBC00},
		{2, 0x4000},
		{0.5, 0x3800},
		{float32(math.Inf(1)), 0x7C00},
		{float32(math.Inf(-1)), 0xFC00},
		{float32(math.Ldexp(1, -24)), 0x0001}, // smallest subnormal
	}
	for _, c := range cases {
		if got := F32ToF16(c.f); got != c.want {
			t.Errorf("F32ToF16(%v) = 0x%04x, want 0x%04x", c.f, got, c.want)
		}
	}
}

func TestF32ToF16FiniteIdempotent(t *testing.T) {
	for u := 0; u <= 0xffff; u++ {
		bits := uint16(u)
		f := F16ToF32(bits)
		if math.IsNaN(float64(f)) {
			continue
		}
		got := F32ToF16(f)
		if got != bits {
			t.Fatalf("F32ToF16(F16ToF32(0x%04x)) = 0x%04x", bits, got)
		}
	}
}

func TestBF16ToF32(t *testing.T) {
	// bf16 is the high 16 bits of a float32; round-trip the top half.
	for _, f := range []float32{0, 1, -1, 2, 0.5, 1234.5} {
		hi := uint16(math.Float32bits(f) >> 16)
		if got := BF16ToF32(hi); math.Float32bits(got) != (uint32(hi) << 16) {
			t.Errorf("BF16ToF32(0x%04x) bits = %08x", hi, math.Float32bits(got))
		}
	}
}
