package simd

import (
	"math"
	"math/rand"
	"testing"
)

func TestExpF32ToChecked(t *testing.T) {
	src := []float32{-2, -0.5, 0, 1, 3}
	dst := make([]float32, len(src))
	if !ExpF32To(dst, src) {
		t.Fatal("ExpF32To returned false for valid input")
	}
	for i, x := range src {
		want := expF32Scalar(x)
		if got := dst[i]; got != want {
			t.Fatalf("index %d: got %v want %v", i, got, want)
		}
	}
	if ExpF32To(nil, nil) {
		t.Fatal("ExpF32To accepted nil input")
	}
	if ExpF32To(make([]float32, len(src)+1), src) {
		t.Fatal("ExpF32To accepted mismatched lengths")
	}
	backing := []float32{1, 2, 3, 4}
	overlapSrc := backing[:3]
	overlapDst := backing[1:4]
	before := append([]float32(nil), overlapDst...)
	if ExpF32To(overlapDst, overlapSrc) {
		t.Fatal("ExpF32To accepted partial overlap")
	}
	for i := range before {
		if overlapDst[i] != before[i] {
			t.Fatalf("partial-overlap rejection mutated dst: got %v want %v", overlapDst, before)
		}
	}
}

func TestExpF32ToInPlace(t *testing.T) {
	x := []float32{-10, -1, -0, 0, 1, 10, 31.5}
	want := make([]float32, len(x))
	copy(want, x)
	for i, v := range want {
		want[i] = expF32Scalar(v)
	}
	if !ExpF32To(x, x) {
		t.Fatal("ExpF32To returned false for in-place input")
	}
	for i := range want {
		if x[i] != want[i] {
			t.Fatalf("index %d: got %v want %v", i, x[i], want[i])
		}
	}
}

func TestExpF32ToFiniteAccuracy(t *testing.T) {
	cases := make([]float32, 0, 20001)
	for i := -8000; i <= 8000; i++ {
		cases = append(cases, float32(i)/100)
	}
	rng := rand.New(rand.NewSource(1))
	for i := 0; i < 20000; i++ {
		cases = append(cases, rng.Float32()*160-80)
	}
	dst := make([]float32, len(cases))
	if !ExpF32To(dst, cases) {
		t.Fatal("ExpF32To returned false for finite accuracy input")
	}
	const tol = 2e-6
	for i, x := range cases {
		want := expF32Scalar(x)
		got := dst[i]
		rel := math.Abs(float64(got-want) / float64(want))
		if rel >= tol || math.IsNaN(rel) {
			t.Fatalf("x=%g got=%g want=%g rel=%g >= %g", x, got, want, rel, tol)
		}
	}
}

func TestExpF32ToExceptionalClassification(t *testing.T) {
	inputs := []float32{
		float32(math.NaN()),
		float32(math.Inf(1)),
		float32(math.Inf(-1)),
		float32(math.Copysign(0, -1)),
		0,
		-150,
		120,
	}
	got := make([]float32, len(inputs))
	if !ExpF32To(got, inputs) {
		t.Fatal("ExpF32To returned false for exceptional input")
	}
	for i, in := range inputs {
		want := expF32Scalar(in)
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

func TestExpF32ToNoAllocs(t *testing.T) {
	src := make([]float32, 1024)
	dst := make([]float32, len(src))
	for i := range src {
		src[i] = float32(i%17) - 8
	}
	if allocs := testing.AllocsPerRun(10, func() {
		if !ExpF32To(dst, src) {
			panic("ExpF32To returned false")
		}
	}); allocs != 0 {
		t.Fatalf("allocs %g", allocs)
	}
}

func BenchmarkExpF32To1024(b *testing.B) {
	src := make([]float32, 1024)
	dst := make([]float32, len(src))
	for i := range src {
		src[i] = float32(i%29-14) * 0.5
	}
	bench := func(b *testing.B, fn func([]float32, []float32)) {
		b.ReportAllocs()
		b.SetBytes(int64(len(src) * 8))
		for range b.N {
			fn(dst, src)
		}
	}
	b.Run("scalar", func(b *testing.B) { bench(b, expF32ToScalar) })
	b.Run("dispatch", func(b *testing.B) {
		bench(b, func(dst, src []float32) {
			if !ExpF32To(dst, src) {
				b.Fatal("ExpF32To returned false")
			}
		})
	})
}

func TestExpF32SIMDInPlaceTailsAndFallback(t *testing.T) {
	for _, n := range []int{8, 9, 15, 16, 25, 40} {
		x := make([]float32, n)
		want := make([]float32, n)
		for i := range x {
			x[i] = float32(i%11 - 5)
		}
		if n >= 25 {
			x[8] = float32(math.Inf(-1))
			x[17] = -33
			x[18] = 33
		}
		for i := range x {
			want[i] = expF32Scalar(x[i])
		}
		if !ExpF32To(x, x) {
			t.Fatal("inplace failed")
		}
		for i := range x {
			if want[i] == 0 {
				if x[i] != 0 {
					t.Fatal("expected zero")
				}
			} else if math.Abs(float64(x[i]-want[i]))/float64(want[i]) >= 2e-6 {
				t.Fatalf("n=%d i=%d got=%g want=%g", n, i, x[i], want[i])
			}
		}
	}
}
