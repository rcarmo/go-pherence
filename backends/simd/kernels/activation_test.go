package kernels

import (
	"math"
	"testing"
)

func TestSiLUAndSiLUMulGolden(t *testing.T) {
	a := []float32{-4, -1, 0, 1, 4}
	b := []float32{2, -3, 4, -5, 6}
	silu := make([]float32, len(a))
	SiLU(silu, a)
	mul := make([]float32, len(a))
	SiLUMul(mul, a, b)
	for i, x := range a {
		want := x / (1 + float32(math.Exp(float64(-x))))
		if math.Abs(float64(silu[i]-want)) > ActivationReferenceTolerance {
			t.Fatalf("SiLU[%d]=%g want %g", i, silu[i], want)
		}
		if math.Abs(float64(mul[i]-want*b[i])) > ActivationReferenceTolerance {
			t.Fatalf("SiLUMul[%d]=%g want %g", i, mul[i], want*b[i])
		}
	}
}

func TestGELUExactAndMulGolden(t *testing.T) {
	a := []float32{-3, -1, 0, 1, 3}
	b := []float32{1, 2, 3, 4, 5}
	gelu := make([]float32, len(a))
	GELUExact(gelu, a)
	mul := make([]float32, len(a))
	GELUExactMul(mul, a, b)
	for i, x := range a {
		want := 0.5 * x * (1 + float32(math.Erf(float64(x)*0.70710678118654752440)))
		if math.Abs(float64(gelu[i]-want)) > ActivationReferenceTolerance {
			t.Fatalf("GELUExact[%d]=%g want %g", i, gelu[i], want)
		}
		if math.Abs(float64(mul[i]-want*b[i])) > ActivationReferenceTolerance {
			t.Fatalf("GELUExactMul[%d]=%g want %g", i, mul[i], want*b[i])
		}
	}
}

func TestGELUTanhAndMulGolden(t *testing.T) {
	a := []float32{-3, -1, 0, 1, 3}
	b := []float32{1, 2, 3, 4, 5}
	gelu := make([]float32, len(a))
	GELUTanh(gelu, a)
	mul := make([]float32, len(a))
	GELUTanhMul(mul, a, b)
	for i, x := range a {
		x3 := x * x * x
		inner := float32(0.7978845608) * (x + 0.044715*x3)
		want := 0.5 * x * (1 + float32(math.Tanh(float64(inner))))
		if math.Abs(float64(gelu[i]-want)) > ActivationReferenceTolerance {
			t.Fatalf("GELUTanh[%d]=%g want %g", i, gelu[i], want)
		}
		if math.Abs(float64(mul[i]-want*b[i])) > ActivationReferenceTolerance {
			t.Fatalf("GELUTanhMul[%d]=%g want %g", i, mul[i], want*b[i])
		}
	}
}

func TestActivationKernelsBoundMalformedInputs(t *testing.T) {
	dst := []float32{99, 99}
	SiLU(dst, []float32{0})
	if dst[0] != 0 || dst[1] != 99 {
		t.Fatalf("SiLU bounded dst=%v", dst)
	}
	dst = []float32{99, 99}
	SiLUMul(dst, []float32{1}, []float32{2, 3})
	if dst[1] != 99 {
		t.Fatalf("SiLUMul mutated tail dst=%v", dst)
	}
	dst = []float32{99, 99}
	GELUTanh(dst, []float32{0})
	if dst[0] != 0 || dst[1] != 99 {
		t.Fatalf("GELUTanh bounded dst=%v", dst)
	}
	dst = []float32{99, 99}
	GELUTanhMul(dst, []float32{0}, []float32{2, 3})
	if dst[0] != 0 || dst[1] != 99 {
		t.Fatalf("GELUTanhMul bounded dst=%v", dst)
	}
}
