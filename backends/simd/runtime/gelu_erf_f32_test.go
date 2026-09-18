package simd

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func geluErfF32ApproxEqual(got, want float32) bool {
	if math.IsNaN(float64(want)) {
		return math.IsNaN(float64(got))
	}
	if math.Float32bits(got) == math.Float32bits(want) {
		return true
	}
	if math.IsInf(float64(want), 0) {
		return false
	}
	return math.Abs(float64(got-want)) <= 2e-6
}

func TestGELUErfF32ToChecked(t *testing.T) {
	src := []float32{-2, -0.5, float32(math.Copysign(0, -1)), 0, 1, 3}
	dst := make([]float32, len(src))
	if !GELUErfF32To(dst, src) {
		t.Fatal("GELUErfF32To returned false for valid input")
	}
	for i, x := range src {
		want := geluErfF32Scalar(x)
		if !geluErfF32ApproxEqual(dst[i], want) {
			t.Fatalf("index %d: got=%g want=%g", i, dst[i], want)
		}
	}
	if GELUErfF32To(nil, nil) {
		t.Fatal("GELUErfF32To accepted nil input")
	}
	if GELUErfF32To(make([]float32, len(src)+1), src) {
		t.Fatal("GELUErfF32To accepted mismatched lengths")
	}
	backing := []float32{1, 2, 3, 4}
	overlapSrc := backing[:3]
	overlapDst := backing[1:4]
	before := append([]float32(nil), overlapDst...)
	if GELUErfF32To(overlapDst, overlapSrc) {
		t.Fatal("GELUErfF32To accepted partial overlap")
	}
	for i := range before {
		if overlapDst[i] != before[i] {
			t.Fatalf("partial-overlap rejection mutated dst: got %v want %v", overlapDst, before)
		}
	}
}

func TestGELUErfF32ToExceptionalClassification(t *testing.T) {
	inputs := []float32{
		float32(math.NaN()),
		float32(math.Inf(1)),
		float32(math.Inf(-1)),
		float32(math.Copysign(0, -1)),
		0,
		-float32(0x1p-20),
		float32(0x1p-20),
		-9,
		9,
	}
	got := make([]float32, len(inputs))
	if !GELUErfF32To(got, inputs) {
		t.Fatal("GELUErfF32To returned false for exceptional input")
	}
	for i, in := range inputs {
		want := geluErfF32Scalar(in)
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

func TestGELUErfF32ToInPlaceSIMDAndTails(t *testing.T) {
	for _, n := range []int{1, 2, 7, 8, 9, 15, 16, 17, 25, 40} {
		x := make([]float32, n)
		want := make([]float32, n)
		for i := range x {
			switch i % 9 {
			case 0:
				x[i] = -7.75 + float32(i%5)*0.25
			case 1:
				x[i] = -2 + float32(i%7)*0.5
			case 2:
				x[i] = float32(i%11-5) * 0.375
			case 3:
				x[i] = 7.5 - float32(i%3)*0.5
			case 4:
				x[i] = -float32(0x1p-20)
			case 5:
				x[i] = float32(math.Copysign(0, -1))
			case 6:
				x[i] = float32(math.Inf(1))
			case 7:
				x[i] = -9
			default:
				x[i] = 9
			}
			want[i] = geluErfF32Scalar(x[i])
		}
		if !GELUErfF32To(x, x) {
			t.Fatal("GELUErfF32To returned false for in-place input")
		}
		for i := range x {
			if !geluErfF32ApproxEqual(x[i], want[i]) {
				t.Fatalf("n=%d i=%d got=%g want=%g bits got=0x%08x want=0x%08x", n, i, x[i], want[i], math.Float32bits(x[i]), math.Float32bits(want[i]))
			}
		}
	}
}

func TestGELUErfF32ToDenseAccuracy(t *testing.T) {
	cases := make([]float32, 0, 600001)
	for i := -200000; i <= 200000; i++ {
		cases = append(cases, float32(i)/25000)
	}
	rng := rand.New(rand.NewSource(1))
	for i := 0; i < 200000; i++ {
		cases = append(cases, rng.Float32()*16-8)
	}
	dst := make([]float32, len(cases))
	if !GELUErfF32To(dst, cases) {
		t.Fatal("GELUErfF32To returned false for dense accuracy input")
	}
	var maxAbs float64
	var worstX, worstGot, worstWant float32
	for i, x := range cases {
		want := geluErfF32Scalar(x)
		got := dst[i]
		abs := math.Abs(float64(got - want))
		if abs > maxAbs {
			maxAbs = abs
			worstX, worstGot, worstWant = x, got, want
		}
		if abs > 2e-6 {
			t.Fatalf("x=%g got=%g want=%g abs=%g", x, got, want, abs)
		}
	}
	t.Logf("max abs err=%g at x=%g (got=%g want=%g, HasVecAsm=%v)", maxAbs, worstX, worstGot, worstWant, HasVecAsm)
}

func TestGELUErfF32ToThresholdBoundaries(t *testing.T) {
	vals := []float32{
		math.Nextafter32(float32(0x1p-12), 0),
		float32(0x1p-12),
		math.Nextafter32(float32(0x1p-12), float32(math.Inf(1))),
		-math.Nextafter32(float32(0x1p-12), 0),
		-float32(0x1p-12),
		-math.Nextafter32(float32(0x1p-12), float32(math.Inf(1))),
		math.Nextafter32(8, 0),
		8,
		math.Nextafter32(8, float32(math.Inf(1))),
		-math.Nextafter32(8, 0),
		-8,
		-math.Nextafter32(8, float32(math.Inf(1))),
	}
	src := make([]float32, 0, len(vals)*8)
	for _, v := range vals {
		for lane := 0; lane < 8; lane++ {
			src = append(src, v)
		}
	}
	got := make([]float32, len(src))
	if !GELUErfF32To(got, src) {
		t.Fatal("GELUErfF32To returned false for threshold input")
	}
	for i, x := range src {
		want := geluErfF32Scalar(x)
		if math.Float32bits(x)&0x7fffffff < geluErfF32SimdAbsMinBits || math.Float32bits(x)&0x7fffffff > geluErfF32SimdAbsMaxBits {
			if math.Float32bits(got[i]) != math.Float32bits(want) {
				t.Fatalf("boundary fallback x=%g got bits 0x%08x want 0x%08x", x, math.Float32bits(got[i]), math.Float32bits(want))
			}
			continue
		}
		if !geluErfF32ApproxEqual(got[i], want) {
			t.Fatalf("boundary simd x=%g got=%g want=%g", x, got[i], want)
		}
	}
}

func TestGELUErfF32ToNoAllocs(t *testing.T) {
	src := make([]float32, 1024)
	dst := make([]float32, len(src))
	for i := range src {
		switch i % 6 {
		case 0:
			src[i] = -7.75 + float32(i%5)*0.25
		case 1:
			src[i] = float32(i%13-6) * 0.5
		case 2:
			src[i] = 7.5 - float32(i%3)*0.5
		case 3:
			src[i] = -float32(0x1p-20)
		case 4:
			src[i] = -9
		default:
			src[i] = 9
		}
	}
	if allocs := testing.AllocsPerRun(10, func() {
		if !GELUErfF32To(dst, src) {
			panic("GELUErfF32To returned false")
		}
	}); allocs != 0 {
		t.Fatalf("allocs %g", allocs)
	}
	if allocs := testing.AllocsPerRun(10, func() {
		if !GELUErfF32To(src, src) {
			panic("GELUErfF32To returned false")
		}
	}); allocs != 0 {
		t.Fatalf("in-place allocs %g", allocs)
	}
}

func benchmarkGELUErfF32To(b *testing.B, n int, name string, fill func([]float32)) {
	src := make([]float32, n)
	dst := make([]float32, len(src))
	fill(src)
	bench := func(b *testing.B, fn func([]float32, []float32)) {
		b.ReportAllocs()
		b.SetBytes(int64(len(src) * 8))
		for range b.N {
			fn(dst, src)
		}
	}
	b.Run(fmt.Sprintf("%s/scalar", name), func(b *testing.B) {
		bench(b, geluErfF32ToScalar)
	})
	b.Run(fmt.Sprintf("%s/dispatch", name), func(b *testing.B) {
		bench(b, func(dst, src []float32) {
			if !GELUErfF32To(dst, src) {
				b.Fatal("GELUErfF32To returned false")
			}
		})
	})
}

func BenchmarkGELUErfF32To1024(b *testing.B) {
	benchmarkGELUErfF32To(b, 1024, "safe", func(src []float32) {
		for i := range src {
			src[i] = -7.75 + float32(i%63)*0.25
		}
	})
	benchmarkGELUErfF32To(b, 1024, "mixed", func(src []float32) {
		for i := range src {
			switch i % 8 {
			case 0:
				src[i] = -7.75 + float32(i%5)*0.25
			case 1:
				src[i] = float32(i%13-6) * 0.5
			case 2:
				src[i] = 7.5 - float32(i%3)*0.5
			case 3:
				src[i] = -float32(0x1p-20)
			case 4:
				src[i] = float32(math.Copysign(0, -1))
			case 5:
				src[i] = -9
			case 6:
				src[i] = 9
			default:
				src[i] = float32(math.Inf(1))
			}
		}
	})
}

func BenchmarkGELUErfF32To3072(b *testing.B) {
	benchmarkGELUErfF32To(b, 3072, "safe", func(src []float32) {
		for i := range src {
			src[i] = -7.75 + float32(i%63)*0.25
		}
	})
	benchmarkGELUErfF32To(b, 3072, "mixed", func(src []float32) {
		for i := range src {
			switch i % 8 {
			case 0:
				src[i] = -7.75 + float32(i%5)*0.25
			case 1:
				src[i] = float32(i%13-6) * 0.5
			case 2:
				src[i] = 7.5 - float32(i%3)*0.5
			case 3:
				src[i] = -float32(0x1p-20)
			case 4:
				src[i] = float32(math.Copysign(0, -1))
			case 5:
				src[i] = -9
			case 6:
				src[i] = 9
			default:
				src[i] = float32(math.Inf(1))
			}
		}
	})
}

func TestGELUErfF32ReductionBoundaries(t *testing.T) {
	var src []float32
	for n := 0; n < 46; n++ {
		center := float32(math.Sqrt(2 * (float64(n) + 0.5) * math.Ln2))
		for _, sign := range []float32{-1, 1} {
			x := sign * center
			for _, v := range []float32{math.Nextafter32(x, float32(math.Inf(-1))), x, math.Nextafter32(x, float32(math.Inf(1)))} {
				for lane := 0; lane < 8; lane++ {
					src = append(src, v)
				}
			}
		}
	}
	dst := make([]float32, len(src))
	GELUErfF32To(dst, src)
	for i, v := range src {
		want := geluErfF32Scalar(v)
		if d := math.Abs(float64(dst[i]) - float64(want)); d > 2e-6 || math.IsNaN(d) {
			t.Fatalf("x%g got%g want%g", v, dst[i], want)
		}
	}
}
