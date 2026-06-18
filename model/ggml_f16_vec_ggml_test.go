//go:build ggml && cgo && linux

package model

import (
	"math"
	"testing"

	"github.com/rcarmo/go-pherence/backends/ggmlcompute"
	"github.com/rcarmo/go-pherence/half"
)

func TestGGMLF16VecDotX86OracleAgainstGGML(t *testing.T) {
	q := make([]uint16, 257)
	k := make([]uint16, 257)
	kF32 := make([]float32, 257)
	for i := range q {
		q[i] = half.F32ToF16(float32(math.Sin(float64(i)*0.019) * 3))
		kF32[i] = half.F16ToF32(half.F32ToF16(float32(math.Cos(float64(i)*0.041) * 4)))
		k[i] = half.F32ToF16(kF32[i])
	}
	got, err := ggmlcompute.VecDotF16(q, k)
	if err != nil {
		t.Fatal(err)
	}
	want := ggmlF16VecDotX86(q, kF32, len(q))
	diff := math.Abs(float64(got - want))
	t.Logf("Go x86 F16 vecdot oracle ggml=%g go=%g diff=%g", got, want, diff)
	if diff > 1e-6 {
		t.Fatalf("vecdot diff=%g", diff)
	}
}

func TestGGMLF16VecX86OracleAgainstGGML(t *testing.T) {
	y := make([]uint16, 257)
	x := make([]uint16, 257)
	for i := range y {
		y[i] = half.F32ToF16(float32(math.Sin(float64(i)*0.031) * 7))
		x[i] = half.F32ToF16(float32(math.Cos(float64(i)*0.047) * 5))
	}
	gotScale := append([]uint16(nil), y...)
	if err := ggmlcompute.VecScaleF16(gotScale, 0.37); err != nil {
		t.Fatal(err)
	}
	goScale := append([]uint16(nil), y...)
	ggmlF16VecScaleX86(goScale, 0.37)
	gotMad := append([]uint16(nil), y...)
	if err := ggmlcompute.VecMadF16(gotMad, x, -0.41); err != nil {
		t.Fatal(err)
	}
	goMad := append([]uint16(nil), y...)
	ggmlF16VecMadX86(goMad, x, -0.41)
	var scaleDiffs, madDiffs int
	for i := range y {
		if gotScale[i] != goScale[i] {
			scaleDiffs++
		}
		if gotMad[i] != goMad[i] {
			madDiffs++
		}
	}
	for i := 0; i < len(y) && i < 8; i++ {
		if gotScale[i] != goScale[i] || gotMad[i] != goMad[i] {
			t.Logf("lane %d scale ggml=%04x go=%04x y=%04x yf=%g; mad ggml=%04x go=%04x x=%04x", i, gotScale[i], goScale[i], y[i], half.F16ToF32(y[i]), gotMad[i], goMad[i], x[i])
		}
	}
	t.Logf("Go x86 F16 vec oracle vs ggml scale diffs=%d mad diffs=%d", scaleDiffs, madDiffs)
	if scaleDiffs != 0 {
		t.Fatalf("scale diffs=%d", scaleDiffs)
	}
	if madDiffs == 0 {
		t.Log("mad oracle already matches ggml")
	}
}
