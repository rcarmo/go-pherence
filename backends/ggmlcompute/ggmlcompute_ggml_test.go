//go:build ggml && cgo && linux

package ggmlcompute

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/rcarmo/go-pherence/half"
)

func TestVecDotF16DiffersFromScalarReduction(t *testing.T) {
	const n = 256
	x := make([]uint16, n)
	y := make([]uint16, n)
	for i := 0; i < n; i++ {
		x[i] = half.F32ToF16(float32(math.Sin(float64(i)*0.031)) * 2)
		y[i] = half.F32ToF16(float32(math.Cos(float64(i)*0.047)) * 3)
	}
	got, err := VecDotF16(x, y)
	if err != nil {
		t.Fatal(err)
	}
	var scalar float32
	for i := 0; i < n; i++ {
		scalar += half.F16ToF32(x[i]) * half.F16ToF32(y[i])
	}
	t.Logf("ggml F16 vecdot=%g scalar=%g delta=%g", got, scalar, got-scalar)
	if math.Abs(float64(got-scalar)) < 1e-6 {
		t.Fatal("ggml F16 vecdot unexpectedly matched scalar reduction exactly")
	}
}

func TestVecDotRowsDirectInitializesCPUTables(t *testing.T) {
	x := make([]float32, 256)
	for i := range x {
		x[i] = float32(math.Sin(float64(i)*0.13)) * 0.5
	}
	q8, err := QuantizeQ8K(x)
	if err != nil {
		t.Fatal(err)
	}
	raw := make([]byte, 210)
	for i := 192; i < 208; i++ {
		raw[i] = 1
	}
	binary.LittleEndian.PutUint16(raw[208:], half.F32ToF16(0.25))
	out := []float32{0}
	if err := VecDotRowsDirect(Q6K, out, raw, 210, q8, 256, 1); err != nil {
		t.Fatal(err)
	}
	if out[0] == 0 {
		t.Fatal("Q6_K/Q8_K direct vecdot returned zero; ggml CPU lookup tables may not be initialized")
	}
}

func TestMulMatF16F32RoundsInputToF16(t *testing.T) {
	const inDim, outDim = 64, 7
	wF16 := make([]uint16, inDim*outDim)
	x := make([]float32, inDim)
	for i := range x {
		x[i] = float32(math.Sin(float64(i)*0.17)) * 1.75
	}
	for r := 0; r < outDim; r++ {
		for c := 0; c < inDim; c++ {
			v := float32(math.Cos(float64(r*13+c)*0.07)) * 0.5
			wF16[r*inDim+c] = half.F32ToF16(v)
		}
	}
	got := make([]float32, outDim)
	if err := MulMatF16F32(got, wF16, x, inDim, outDim); err != nil {
		t.Fatal(err)
	}
	wantRoundedInput := make([]float32, outDim)
	wantF32Input := make([]float32, outDim)
	for r := 0; r < outDim; r++ {
		var sumRounded, sumF32 float32
		for c := 0; c < inDim; c++ {
			w := half.F16ToF32(wF16[r*inDim+c])
			xRounded := half.F16ToF32(half.F32ToF16(x[c]))
			sumRounded += w * xRounded
			sumF32 += w * x[c]
		}
		wantRoundedInput[r] = sumRounded
		wantF32Input[r] = sumF32
	}
	var sawUnroundedDiff bool
	for i := range got {
		if d := math.Abs(float64(got[i] - wantRoundedInput[i])); d > 1e-5 {
			t.Fatalf("row %d ggml F16 mul_mat=%g rounded-input scalar=%g diff=%g", i, got[i], wantRoundedInput[i], d)
		}
		if math.Abs(float64(got[i]-wantF32Input[i])) > 1e-5 {
			sawUnroundedDiff = true
		}
	}
	if !sawUnroundedDiff {
		t.Fatal("ggml F16 mul_mat unexpectedly matched unrounded F32-input accumulation")
	}
}
