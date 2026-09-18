package simd

import (
	"math"
	"math/big"
	"math/rand"
	"reflect"
	"testing"
)

func affineOracle(a, b, c float32) float32 {
	x := new(big.Float).SetPrec(1024).SetFloat64(float64(a))
	y := new(big.Float).SetPrec(1024).SetFloat64(float64(b))
	z := new(big.Float).SetPrec(1024).SetFloat64(float64(c))
	x.Mul(x, y)
	x.Add(x, z)
	result, _ := x.Float32()
	return result
}
func TestAffineF32ExactTailsAndCanaries(t *testing.T) {
	t.Logf("AVX2/FMA dispatch available: %v", HasAffineF32Asm())
	scales := [][2]float32{{1, 0}, {-1, float32(math.Copysign(0, -1))}, {1.5, -math.SmallestNonzeroFloat32}, {.5, math.SmallestNonzeroFloat32}, {.731, -.013}, {math.MaxFloat32, 0}}
	for offset := 0; offset < 8; offset++ {
		for n := 0; n <= 67; n++ {
			for _, pair := range scales {
				data := make([]float32, offset+n+9)
				for i := range data {
					data[i] = 9123
				}
				x := data[offset : offset+n]
				for i := range x {
					x[i] = float32(i-31) * .125
				}
				if n > 0 {
					x[0] = math.Float32frombits(0x3f800001)
				}
				if n > 1 {
					x[n-1] = math.SmallestNonzeroFloat32
				}
				before := append([]float32(nil), data...)
				if !AffineF32InPlaceChecked(x, pair[0], pair[1]) {
					t.Fatal("rejected finite inputs")
				}
				for i, v := range x {
					want := affineOracle(before[offset+i], pair[0], pair[1])
					if math.Float32bits(v) != math.Float32bits(want) {
						t.Fatalf("offset%d n%d i%d got%08x want%08x", offset, n, i, math.Float32bits(v), math.Float32bits(want))
					}
				}
				for i, v := range data {
					if (i < offset || i >= offset+n) && v != before[i] {
						t.Fatal("canary overwritten")
					}
				}
			}
		}
	}
}
func TestAffineF32RandomAndSignedZero(t *testing.T) {
	rng := rand.New(rand.NewSource(51237))
	checks := 0
	for batch := 0; batch < 300; batch++ {
		scale, shift := math.Float32frombits(rng.Uint32()), math.Float32frombits(rng.Uint32())
		if !affineFinite(scale) || !affineFinite(shift) {
			continue
		}
		values := make([]float32, 65)
		for i := range values {
			for {
				values[i] = math.Float32frombits(rng.Uint32())
				if affineFinite(values[i]) {
					break
				}
			}
		}
		before := append([]float32(nil), values...)
		if !AffineF32InPlaceChecked(values, scale, shift) {
			t.Fatal("random finite rejected")
		}
		for i, v := range values {
			want := affineOracle(before[i], scale, shift)
			fallback := FMA32Scalar(before[i], scale, shift)
			if math.Float32bits(v) != math.Float32bits(want) || math.Float32bits(fallback) != math.Float32bits(want) {
				t.Fatalf("FMA mismatch %08x*%08x+%08x got%08x scalar%08x want%08x", math.Float32bits(before[i]), math.Float32bits(scale), math.Float32bits(shift), math.Float32bits(v), math.Float32bits(fallback), math.Float32bits(want))
			}
			checks++
		}
	}
	for _, a := range []float32{0, float32(math.Copysign(0, -1))} {
		for _, b := range []float32{1, -1} {
			for _, c := range []float32{0, float32(math.Copysign(0, -1))} {
				values := make([]float32, 9)
				for i := range values {
					values[i] = a
				}
				if !AffineF32InPlaceChecked(values, b, c) {
					t.Fatal("zeros rejected")
				}
				want := affineOracle(a, b, c)
				for _, v := range values {
					if math.Float32bits(v) != math.Float32bits(want) {
						t.Fatal("zero sign")
					}
				}
			}
		}
	}
	t.Logf("random exact big.Float comparisons: %d", checks)
}
func TestAffineF32ValidationAndAllocation(t *testing.T) {
	for _, bad := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
		for _, where := range []string{"scale", "shift", "data"} {
			x := []float32{1, 2, 3}
			scale, shift := float32(2), float32(3)
			switch where {
			case "scale":
				scale = bad
			case "shift":
				shift = bad
			case "data":
				x[2] = bad
			}
			before := append([]float32(nil), x...)
			if AffineF32InPlaceChecked(x, scale, shift) {
				t.Fatal("invalid accepted")
			}
			for i, v := range x {
				if math.Float32bits(v) != math.Float32bits(before[i]) {
					t.Fatal("rejected call wrote input")
				}
			}
		}
	}
	if !AffineF32InPlaceChecked(nil, 1, 0) {
		t.Fatal("empty rejected")
	}
	x := make([]float32, 4096)
	if allocations := testing.AllocsPerRun(20, func() {
		if !AffineF32InPlaceChecked(x, 1, 0) {
			panic("affine failed")
		}
	}); allocations != 0 {
		t.Fatal("allocations", allocations)
	}
	expected := []float32{3, 5, 7}
	x = []float32{1, 2, 3}
	affineF32Scalar(x, 2, 1)
	if !reflect.DeepEqual(x, expected) {
		t.Fatal("fallback")
	}
}
func BenchmarkAffineF32(b *testing.B) {
	for _, n := range []int{43, 129, 4096, 160000} {
		for _, mode := range []string{"scalar", "checked"} {
			b.Run(fmtAffineBench(n, mode), func(b *testing.B) {
				x := make([]float32, n)
				for i := range x {
					x[i] = float32(i%137) / 137
				}
				source := append([]float32(nil), x...)
				b.ReportAllocs()
				b.SetBytes(int64(n * 8))
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					copy(x, source) // identical non-converging inputs in both arms
					if mode == "scalar" {
						if !affineFinite(.999) || !affineFinite(.0002) {
							b.Fatal("invalid")
						}
						for _, v := range x {
							if !affineFinite(v) {
								b.Fatal("nonfinite")
							}
						}
						affineF32Scalar(x, .999, .0002)
					} else {
						if !AffineF32InPlaceChecked(x, .999, .0002) {
							b.Fatal("failed")
						}
					}
				}
			})
		}
	}
}
func fmtAffineBench(n int, mode string) string { // avoid reflection in timed loop
	switch n {
	case 43:
		return "43/" + mode
	case 129:
		return "129/" + mode
	case 4096:
		return "4096/" + mode
	default:
		return "160000/" + mode
	}
}
