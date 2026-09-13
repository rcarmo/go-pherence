package simd

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func TestSinF32ToChecked(t *testing.T) {
	src := []float32{-2, -0.5, -0, 0, 1, 3}
	dst := make([]float32, len(src))
	if !SinF32To(dst, src) {
		t.Fatal("SinF32To returned false for valid input")
	}
	for i, x := range src {
		want := sinF32Scalar(x)
		if got := dst[i]; math.Float32bits(got) != math.Float32bits(want) {
			t.Fatalf("index %d: got bits 0x%08x want 0x%08x", i, math.Float32bits(got), math.Float32bits(want))
		}
	}
	if SinF32To(nil, nil) {
		t.Fatal("SinF32To accepted nil input")
	}
	if SinF32To(make([]float32, len(src)+1), src) {
		t.Fatal("SinF32To accepted mismatched lengths")
	}
	backing := []float32{1, 2, 3, 4}
	overlapSrc := backing[:3]
	overlapDst := backing[1:4]
	before := append([]float32(nil), overlapDst...)
	if SinF32To(overlapDst, overlapSrc) {
		t.Fatal("SinF32To accepted partial overlap")
	}
	for i := range before {
		if overlapDst[i] != before[i] {
			t.Fatalf("partial-overlap rejection mutated dst: got %v want %v", overlapDst, before)
		}
	}
}

func TestSinF32ToInPlaceTailsAndFallback(t *testing.T) {
	for _, n := range []int{7, 8, 9, 15, 16, 17, 25, 40} {
		x := make([]float32, n)
		want := make([]float32, n)
		for i := range x {
			x[i] = float32(i%19-9) * 0.5
		}
		if n >= 17 {
			x[1] = float32(math.Copysign(0, -1))
			x[8] = 32.5
			x[9] = -32.5
			x[10] = float32(math.Inf(1))
			x[11] = float32(math.NaN())
		}
		for i := range x {
			want[i] = sinF32Scalar(x[i])
		}
		if !SinF32To(x, x) {
			t.Fatal("SinF32To returned false for in-place input")
		}
		for i := range x {
			if math.IsNaN(float64(want[i])) {
				if !math.IsNaN(float64(x[i])) {
					t.Fatalf("n=%d i=%d got %v want NaN", n, i, x[i])
				}
				continue
			}
			if math.Float32bits(want[i]) == math.Float32bits(float32(math.Copysign(0, -1))) || want[i] == 0 {
				if math.Float32bits(x[i]) != math.Float32bits(want[i]) {
					t.Fatalf("n=%d i=%d got bits 0x%08x want 0x%08x", n, i, math.Float32bits(x[i]), math.Float32bits(want[i]))
				}
				continue
			}
			if abs := math.Abs(float64(x[i] - want[i])); abs > 2e-6 {
				t.Fatalf("n=%d i=%d got=%g want=%g abs=%g", n, i, x[i], want[i], abs)
			}
		}
	}
}

func TestSinF32ToExceptionalClassification(t *testing.T) {
	inputs := []float32{
		float32(math.NaN()),
		float32(math.Inf(1)),
		float32(math.Inf(-1)),
		float32(math.Copysign(0, -1)),
		0,
		33,
		-33,
	}
	got := make([]float32, len(inputs))
	if !SinF32To(got, inputs) {
		t.Fatal("SinF32To returned false for exceptional input")
	}
	for i, in := range inputs {
		want := sinF32Scalar(in)
		if math.IsNaN(float64(want)) {
			if !math.IsNaN(float64(got[i])) {
				t.Fatalf("input %v: got %v want NaN", in, got[i])
			}
			continue
		}
		if math.Float32bits(got[i]) != math.Float32bits(want) {
			t.Fatalf("input %v: got bits 0x%08x want 0x%08x (got=%v want=%v)", in, math.Float32bits(got[i]), math.Float32bits(want), got[i], want)
		}
	}
}

func TestSinF32ToFiniteAccuracy(t *testing.T) {
	cases := make([]float32, 0, 900000)
	for i := -320000; i <= 320000; i++ {
		cases = append(cases, float32(i)/10000)
	}
	rng := rand.New(rand.NewSource(1))
	for i := 0; i < 200000; i++ {
		cases = append(cases, rng.Float32()*64-32)
	}
	dst := make([]float32, len(cases))
	if !SinF32To(dst, cases) {
		t.Fatal("SinF32To returned false for finite accuracy input")
	}
	const tol = 2e-6
	var maxAbs float64
	var worstX, worstGot, worstWant float32
	for i, x := range cases {
		want := sinF32Scalar(x)
		got := dst[i]
		abs := math.Abs(float64(got - want))
		if abs > maxAbs {
			maxAbs = abs
			worstX, worstGot, worstWant = x, got, want
		}
		if abs > tol || math.IsNaN(abs) {
			t.Fatalf("x=%g got=%g want=%g abs=%g > %g", x, got, want, abs, tol)
		}
	}
	t.Logf("max abs err=%g at x=%g (got=%g want=%g, HasVecAsm=%v)", maxAbs, worstX, worstGot, worstWant, HasVecAsm)
}

func TestSinF32ToNoAllocs(t *testing.T) {
	src := make([]float32, 1024)
	dst := make([]float32, len(src))
	for i := range src {
		src[i] = float32(i%65-32) * 0.5
	}
	if allocs := testing.AllocsPerRun(10, func() {
		if !SinF32To(dst, src) {
			panic("SinF32To returned false")
		}
	}); allocs != 0 {
		t.Fatalf("allocs %g", allocs)
	}
	if allocs := testing.AllocsPerRun(10, func() {
		if !SinF32To(src, src) {
			panic("SinF32To returned false")
		}
	}); allocs != 0 {
		t.Fatalf("in-place allocs %g", allocs)
	}
}

func BenchmarkSinF32To(b *testing.B) {
	for _, n := range []int{7, 8, 15, 16, 63, 64, 255, 1024} {
		src := make([]float32, n)
		dst := make([]float32, n)
		for i := range src {
			src[i] = float32(i%61-30) * 0.5
		}
		bench := func(b *testing.B, fn func([]float32, []float32)) {
			b.ReportAllocs()
			b.SetBytes(int64(len(src) * 8))
			for range b.N {
				fn(dst, src)
			}
		}
		b.Run(fmt.Sprintf("n=%d/scalar", n), func(b *testing.B) {
			bench(b, sinF32ToScalar)
		})
		b.Run(fmt.Sprintf("n=%d/dispatch", n), func(b *testing.B) {
			bench(b, func(dst, src []float32) {
				if !SinF32To(dst, src) {
					b.Fatal("SinF32To returned false")
				}
			})
		})
	}
}

func TestSinF32QuadrantBoundaries(t *testing.T) {
	var src []float32
	for k := -40; k <= 40; k++ {
		x := float32(float64(k) * math.Pi / 4)
		for _, v := range []float32{math.Nextafter32(x, float32(math.Inf(-1))), x, math.Nextafter32(x, float32(math.Inf(1)))} {
			// Repeat per lane so every boundary reaches SIMD, rather than only tails.
			for lane := 0; lane < 8; lane++ {
				src = append(src, v)
			}
		}
	}
	got, want := make([]float32, len(src)), make([]float32, len(src))
	SinF32To(got, src)
	for i, v := range src {
		want[i] = sinF32Scalar(v)
	}
	for i, v := range got {
		if diff := math.Abs(float64(v) - float64(want[i])); diff > 2e-6 || math.IsNaN(diff) {
			t.Fatalf("x%g got%g want%g", src[i], v, want[i])
		}
	}
}
