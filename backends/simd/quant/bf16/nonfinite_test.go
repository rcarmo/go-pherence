package bf16

import (
	"math"
	"testing"
)

func TestF32ToBF16PreservesNonfiniteClassification(t *testing.T) {
	for _, bits := range []uint32{0x7f800001, 0x7f800100, 0x7fffffff, 0xff800001, 0xffffffff} {
		x := math.Float32frombits(bits)
		got := BF16ToF32(F32ToBF16(x))
		if !math.IsNaN(float64(got)) {
			t.Fatalf("NaN %08x converted to %g", bits, got)
		}
	}
	for _, bits := range []uint32{0, 0x80000000, 0x7f800000, 0xff800000} {
		if got := uint32(F32ToBF16(math.Float32frombits(bits))) << 16; got != bits {
			t.Fatalf("special %08x -> %08x", bits, got)
		}
	}
}
func TestBF16RMSNormInvalidEpsilonDoesNotWrite(t *testing.T) {
	for _, eps := range []float32{-1, float32(math.NaN()), float32(math.Inf(1))} {
		x := []uint16{F32ToBF16(1)}
		w := []uint16{F32ToBF16(1)}
		if BF16RMSNormChecked(x, w, eps) {
			t.Fatal("invalid epsilon accepted", eps)
		}
		if x[0] != F32ToBF16(1) {
			t.Fatal("invalid checked operation wrote")
		}
		BF16RMSNorm(x, w, eps)
		if x[0] != F32ToBF16(1) {
			t.Fatal("invalid void operation wrote")
		}
	}
}
