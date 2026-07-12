package diffusiongemma

import (
	"math"
	"slices"
	"testing"

	"github.com/rcarmo/go-pherence/half"
)

func TestQuantizeDequantQ8_0ForExpertDotMatchesReference(t *testing.T) {
	x := make([]float32, 64)
	for i := range x {
		x[i] = float32((i%17)-8) * 0.03125
	}
	// These two values straddle a code boundary only if the F16-rounded
	// stored scale is incorrectly reused to derive the integer code.
	x[0], x[1] = 72.75873, 55.274055
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
		dRaw := amax / 127
		d := half.F16ToF32(half.F32ToF16(dRaw))
		id := float32(0)
		if dRaw != 0 {
			id = 1 / dRaw
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

	// Guard the distinction from the old implementation, which derived the
	// integer codes from the already-F16-rounded stored scale.
	old := make([]float32, len(x))
	for b := 0; b < len(x); b += 32 {
		amax := float32(0)
		for _, v := range x[b : b+32] {
			amax = max(amax, float32(math.Abs(float64(v))))
		}
		d := half.F16ToF32(half.F32ToF16(amax / 127))
		for i, v := range x[b : b+32] {
			old[b+i] = float32(int8(math.Round(float64(v/d)))) * d
		}
	}
	if slices.Equal(got, old) {
		t.Fatal("fixture does not distinguish ggml F32 code scale from rounded stored scale")
	}
}
