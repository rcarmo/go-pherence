package diffusiongemma

import (
	"math"
	"testing"

	"github.com/rcarmo/go-pherence/half"
)

func TestQuantizeDequantQ8_0ForExpertDotMatchesReference(t *testing.T) {
	x := make([]float32, 64)
	for i := range x {
		x[i] = float32((i%17)-8) * 0.03125
	}
	got := make([]float32, len(x))
	quantizeDequantQ8_0ForExpertDot(got, x)
	want := make([]float32, len(x))
	for b := 0; b < len(x); b += 32 {
		amax := float32(0)
		for _, v := range x[b : b+32] {
			av := v
			if av < 0 {
				av = -av
			}
			if av > amax {
				amax = av
			}
		}
		d := half.F16ToF32(half.F32ToF16(amax / 127))
		id := float32(0)
		if d != 0 {
			id = 1 / d
		}
		for i, v := range x[b : b+32] {
			q := int(math.Round(float64(v * id)))
			if q > 127 {
				q = 127
			}
			if q < -128 {
				q = -128
			}
			want[b+i] = float32(int8(q)) * d
		}
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("idx=%d got=%g want=%g", i, got[i], want[i])
		}
	}
}
