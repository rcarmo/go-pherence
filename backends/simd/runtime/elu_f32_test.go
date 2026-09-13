package simd

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func eluF32ApproxEqual(got, want float32) bool {
	if math.IsNaN(float64(want)) {
		return math.IsNaN(float64(got))
	}
	if math.Float32bits(got) == math.Float32bits(want) {
		return true
	}
	if math.IsInf(float64(want), 0) {
		return false
	}
	abs := math.Abs(float64(got - want))
	if want == 0 {
		return abs == 0
	}
	if abs > 2e-6 {
		return false
	}
	den := math.Abs(float64(want))
	if den >= 1e-5 {
		return abs/den <= 2e-6
	}
	if abs > eluF32NearZeroAbsTol {
		return false
	}
	return got != 0
}

func TestELUF32ToChecked(t *testing.T) {
	src := []float32{-2, -0.5, float32(math.Copysign(0, -1)), 0, 1, 3}
	dst := make([]float32, len(src))
	if !ELUF32To(dst, src) {
		t.Fatal("ELUF32To returned false for valid input")
	}
	for i, x := range src {
		want := eluF32Scalar(x)
		if got := dst[i]; math.Float32bits(got) != math.Float32bits(want) {
			t.Fatalf("index %d: got bits 0x%08x want 0x%08x", i, math.Float32bits(got), math.Float32bits(want))
		}
	}
	if ELUF32To(nil, nil) {
		t.Fatal("ELUF32To accepted nil input")
	}
	if ELUF32To(make([]float32, len(src)+1), src) {
		t.Fatal("ELUF32To accepted mismatched lengths")
	}
	backing := []float32{1, 2, 3, 4}
	overlapSrc := backing[:3]
	overlapDst := backing[1:4]
	before := append([]float32(nil), overlapDst...)
	if ELUF32To(overlapDst, overlapSrc) {
		t.Fatal("ELUF32To accepted partial overlap")
	}
	for i := range before {
		if overlapDst[i] != before[i] {
			t.Fatalf("partial-overlap rejection mutated dst: got %v want %v", overlapDst, before)
		}
	}
}

func TestELUF32ToInPlaceTailsAndFallback(t *testing.T) {
	for _, n := range []int{1, 2, 7, 8, 9, 15, 16, 17, 25, 40} {
		x := make([]float32, n)
		want := make([]float32, n)
		for i := range x {
			switch {
			case i%7 == 0:
				x[i] = float32(i%13) - 6
			case i%7 == 1:
				x[i] = -float32(i+1) * 0.25
			case i%7 == 2:
				x[i] = float32(math.Copysign(0, -1))
			case i%7 == 3:
				x[i] = 0
			case i%7 == 4:
				x[i] = -20
			case i%7 == 5:
				x[i] = -float32(0x1p-54)
			default:
				x[i] = float32(math.Inf(1))
			}
			want[i] = eluF32Scalar(x[i])
		}
		if !ELUF32To(x, x) {
			t.Fatal("ELUF32To returned false for in-place input")
		}
		for i := range x {
			if !eluF32ApproxEqual(x[i], want[i]) {
				t.Fatalf("n=%d i=%d got=%g want=%g bits got=0x%08x want=0x%08x", n, i, x[i], want[i], math.Float32bits(x[i]), math.Float32bits(want[i]))
			}
		}
	}
}

func TestELUF32ToExceptionalClassification(t *testing.T) {
	inputs := []float32{
		float32(math.NaN()),
		float32(math.Inf(1)),
		float32(math.Inf(-1)),
		float32(math.Copysign(0, -1)),
		0,
		-150,
		-20,
		-16,
		-float32(0x1p-54),
		float32(math.Nextafter32(-float32(0x1p-54), float32(math.Inf(-1)))),
		120,
	}
	got := make([]float32, len(inputs))
	if !ELUF32To(got, inputs) {
		t.Fatal("ELUF32To returned false for exceptional input")
	}
	for i, in := range inputs {
		want := eluF32Scalar(in)
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

func TestELUF32ToFiniteAccuracy(t *testing.T) {
	cases := make([]float32, 0, 500000)
	for i := -160000; i <= 160000; i++ {
		cases = append(cases, float32(i)/10000)
	}
	rng := rand.New(rand.NewSource(1))
	for i := 0; i < 200000; i++ {
		cases = append(cases, rng.Float32()*240-120)
	}
	dst := make([]float32, len(cases))
	if !ELUF32To(dst, cases) {
		t.Fatal("ELUF32To returned false for finite accuracy input")
	}
	var maxAbs, maxRel float64
	var worstAbsX, worstRelX, worstAbsGot, worstAbsWant, worstRelGot, worstRelWant float32
	for i, x := range cases {
		want := eluF32Scalar(x)
		got := dst[i]
		if want >= 0 {
			if math.Float32bits(got) != math.Float32bits(want) {
				t.Fatalf("x=%g positive passthrough got bits 0x%08x want 0x%08x", x, math.Float32bits(got), math.Float32bits(want))
			}
			continue
		}
		abs := math.Abs(float64(got - want))
		if abs > maxAbs {
			maxAbs = abs
			worstAbsX, worstAbsGot, worstAbsWant = x, got, want
		}
		if want != 0 {
			rel := abs / math.Abs(float64(want))
			if rel > maxRel {
				maxRel = rel
				worstRelX, worstRelGot, worstRelWant = x, got, want
			}
		}
		if !eluF32ApproxEqual(got, want) {
			t.Fatalf("x=%g got=%g want=%g abs=%g", x, got, want, abs)
		}
	}
	t.Logf("max abs err=%g at x=%g (got=%g want=%g, HasVecAsm=%v)", maxAbs, worstAbsX, worstAbsGot, worstAbsWant, HasVecAsm)
	t.Logf("max rel err=%g at x=%g (got=%g want=%g, HasVecAsm=%v)", maxRel, worstRelX, worstRelGot, worstRelWant, HasVecAsm)
}

func TestELUF32ToNearZeroULPs(t *testing.T) {
	vals := make([]float32, 0, 4096)
	for p := 10; p <= 80; p++ {
		base := -float32(math.Ldexp(1, -p))
		vals = append(vals, base)
		for step := 1; step <= 16; step++ {
			vals = append(vals,
				math.Nextafter32(base, float32(math.Inf(-1))),
				math.Nextafter32(base, 0),
			)
			base = math.Nextafter32(base, float32(math.Inf(-1)))
		}
	}
	// Ensure SIMD coverage by repeating each input across a full AVX2 lane group.
	src := make([]float32, 0, len(vals)*8)
	for _, v := range vals {
		for lane := 0; lane < 8; lane++ {
			src = append(src, v)
		}
	}
	got := make([]float32, len(src))
	if !ELUF32To(got, src) {
		t.Fatal("ELUF32To returned false for near-zero input")
	}
	for i, v := range src {
		want := eluF32Scalar(v)
		if want != 0 && got[i] == 0 {
			t.Fatalf("x=%g collapsed to zero; want=%g", v, want)
		}
		if !eluF32ApproxEqual(got[i], want) {
			abs := math.Abs(float64(got[i] - want))
			t.Fatalf("x=%g got=%g want=%g abs=%g", v, got[i], want, abs)
		}
	}
}

func TestELUF32ToNoAllocs(t *testing.T) {
	src := make([]float32, 1024)
	dst := make([]float32, len(src))
	for i := range src {
		switch i % 5 {
		case 0:
			src[i] = float32(i%31) - 15
		case 1:
			src[i] = -float32(i%17) * 0.5
		case 2:
			src[i] = -float32(0x1p-20)
		case 3:
			src[i] = -20
		default:
			src[i] = 3
		}
	}
	if allocs := testing.AllocsPerRun(10, func() {
		if !ELUF32To(dst, src) {
			panic("ELUF32To returned false")
		}
	}); allocs != 0 {
		t.Fatalf("allocs %g", allocs)
	}
	if allocs := testing.AllocsPerRun(10, func() {
		if !ELUF32To(src, src) {
			panic("ELUF32To returned false")
		}
	}); allocs != 0 {
		t.Fatalf("in-place allocs %g", allocs)
	}
}

func BenchmarkELUF32To(b *testing.B) {
	patterns := map[string]func([]float32){
		"safe-neg": func(src []float32) {
			for i := range src {
				src[i] = -12 + float32(i%23)*0.5
			}
		},
		"mixed": func(src []float32) {
			for i := range src {
				switch i % 6 {
				case 0:
					src[i] = -12 + float32(i%23)*0.5
				case 1:
					src[i] = -float32(i%17) * 0.25
				case 2:
					src[i] = -float32(0x1p-20)
				case 3:
					src[i] = -20
				case 4:
					src[i] = 0
				default:
					src[i] = float32(i%19) * 0.5
				}
			}
		},
	}
	for _, n := range []int{8, 16, 64, 256, 1024} {
		for name, fill := range patterns {
			src := make([]float32, n)
			dst := make([]float32, n)
			fill(src)
			bench := func(b *testing.B, fn func([]float32, []float32)) {
				b.ReportAllocs()
				b.SetBytes(int64(len(src) * 8))
				for range b.N {
					fn(dst, src)
				}
			}
			b.Run(fmt.Sprintf("n=%d/%s/scalar", n, name), func(b *testing.B) {
				bench(b, eluF32ToScalar)
			})
			b.Run(fmt.Sprintf("n=%d/%s/dispatch", n, name), func(b *testing.B) {
				bench(b, func(dst, src []float32) {
					if !ELUF32To(dst, src) {
						b.Fatal("ELUF32To returned false")
					}
				})
			})
		}
	}
}

func TestELUF32TinyAndPositiveLanes(t *testing.T) {
	for _, x := range []float32{-0x1.8p-54, -0x1.1p-52, -0x1.5p-40, -0x1p-20, math.Float32frombits(0x80000000), 0, 1, math.MaxFloat32, float32(math.Inf(1))} {
		src, dst := make([]float32, 24), make([]float32, 24)
		for i := range src {
			src[i] = x
			if i%3 == 0 {
				src[i] = -0.5
			}
		}
		ELUF32To(dst, src)
		for i, v := range src {
			want := eluF32Scalar(v)
			if v >= 0 || v >= -eluF32TinyScalarMax {
				if math.Float32bits(dst[i]) != math.Float32bits(want) {
					t.Fatalf("x%g got%08x want%08x", v, math.Float32bits(dst[i]), math.Float32bits(want))
				}
			}
		}
	}
}

func BenchmarkELUF32ModelMixed(b *testing.B) {
	src, dst := make([]float32, 4096), make([]float32, 4096)
	for i := range src {
		src[i] = float32(i%137-68) / 16
	}
	b.Run("scalar", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			eluF32ToScalar(dst, src)
		}
	})
	b.Run("dispatch", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			ELUF32To(dst, src)
		}
	})
}

func TestELUF32NaNPayloadPreserved(t *testing.T) {
	for _, bits := range []uint32{0x7fc01234, 0xffc05678, 0x7f801234, 0xff801234} {
		src, dst := make([]float32, 25), make([]float32, 25)
		for i := range src {
			src[i] = -0.5
		}
		src[8] = math.Float32frombits(bits)
		src[24] = src[8]
		ELUF32To(dst, src)
		if math.Float32bits(dst[8]) != bits || math.Float32bits(dst[24]) != bits {
			t.Fatalf("payload %08x lost", bits)
		}
	}
}
