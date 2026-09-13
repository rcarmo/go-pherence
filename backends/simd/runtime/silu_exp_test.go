package simd

import (
	"math"
	"math/rand"
	"testing"
)

func siluMulReference(gate, up float32) float32 {
	s := gate / (1 + float32(math.Exp(float64(-gate))))
	return s * up
}

func siluMulApproxEqual(got, want float32) bool {
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
	if abs <= 5e-5 {
		return true
	}
	den := math.Abs(float64(want))
	if den == 0 {
		return false
	}
	return abs/den <= 8e-6
}

func TestSiLUMulExpToChecked(t *testing.T) {
	gate := []float32{-3, -0.5, 0, 1, 4}
	up := []float32{2, -3, 4, -5, 6}
	dst := make([]float32, len(gate))
	scratch := make([]float32, len(gate))
	if !SiLUMulExpTo(dst, gate, up, scratch) {
		t.Fatal("SiLUMulExpTo returned false for valid input")
	}
	for i := range dst {
		want := siluMulReference(gate[i], up[i])
		if !siluMulApproxEqual(dst[i], want) {
			t.Fatalf("distinct[%d]=%g want %g", i, dst[i], want)
		}
	}

	if SiLUMulExpTo(nil, gate, up, scratch) ||
		SiLUMulExpTo(dst, gate[:len(gate)-1], up, scratch) ||
		SiLUMulExpTo(dst, gate, up[:len(up)-1], scratch) ||
		SiLUMulExpTo(dst, gate, up, scratch[:len(scratch)-1]) {
		t.Fatal("SiLUMulExpTo accepted malformed lengths")
	}

	backing := []float32{9, 1, 2, 3, 4, 5, 6, 7, 8, 10}
	overlapGate := backing[1:6]
	overlapDst := backing[2:7]
	overlapUp := []float32{1, 1, 1, 1, 1}
	overlapScratch := []float32{1, 1, 1, 1, 1}
	beforeOverlap := append([]float32(nil), overlapDst...)
	if SiLUMulExpTo(overlapDst, overlapGate, overlapUp, overlapScratch) {
		t.Fatal("SiLUMulExpTo accepted partial dst/gate overlap")
	}
	for i := range beforeOverlap {
		if overlapDst[i] != beforeOverlap[i] {
			t.Fatalf("partial dst/gate rejection mutated dst: got %v want %v", overlapDst, beforeOverlap)
		}
	}

	backing2 := []float32{9, 1, 2, 3, 4, 5, 6, 7, 8, 10}
	overlapDst = backing2[1:6]
	overlapUp = backing2[2:7]
	overlapGate = []float32{1, 2, 3, 4, 5}
	beforeOverlap = append([]float32(nil), overlapDst...)
	if SiLUMulExpTo(overlapDst, overlapGate, overlapUp, overlapScratch) {
		t.Fatal("SiLUMulExpTo accepted partial dst/up overlap")
	}
	for i := range beforeOverlap {
		if overlapDst[i] != beforeOverlap[i] {
			t.Fatalf("partial dst/up rejection mutated dst: got %v want %v", overlapDst, beforeOverlap)
		}
	}

	validDst := make([]float32, len(gate))
	if SiLUMulExpTo(validDst, gate, up, gate) {
		t.Fatal("SiLUMulExpTo accepted scratch aliasing gate")
	}
	if SiLUMulExpTo(validDst, gate, up, validDst) {
		t.Fatal("SiLUMulExpTo accepted scratch aliasing dst")
	}
	scratchOverlapBacking := []float32{1, 2, 3, 4, 5, 6, 7}
	overlapUp = scratchOverlapBacking[:5]
	overlapScratch = scratchOverlapBacking[1:6]
	beforeScratch := append([]float32(nil), overlapScratch...)
	if SiLUMulExpTo(validDst, gate, overlapUp, overlapScratch) {
		t.Fatal("SiLUMulExpTo accepted scratch overlapping up")
	}
	for i := range beforeScratch {
		if overlapScratch[i] != beforeScratch[i] {
			t.Fatalf("scratch overlap rejection mutated scratch: got %v want %v", overlapScratch, beforeScratch)
		}
	}
}

func TestSiLUMulExpToInPlace(t *testing.T) {
	gateBase := []float32{-10, -2, -0, 0, 1, 10, 31.5}
	upBase := []float32{2, -3, 4, -5, 6, -7, 8}

	for _, mode := range []string{"dst==gate", "dst==up", "dst==gate==up"} {
		scratch := make([]float32, len(gateBase))
		want := make([]float32, len(gateBase))
		for i := range want {
			g := gateBase[i]
			u := upBase[i]
			if mode == "dst==gate==up" {
				u = gateBase[i]
			}
			want[i] = siluMulReference(g, u)
		}
		switch mode {
		case "dst==gate":
			gate := append([]float32(nil), gateBase...)
			up := append([]float32(nil), upBase...)
			if !SiLUMulExpTo(gate, gate, up, scratch) {
				t.Fatalf("%s returned false", mode)
			}
			for i := range gate {
				if !siluMulApproxEqual(gate[i], want[i]) {
					t.Fatalf("%s[%d]=%g want %g", mode, i, gate[i], want[i])
				}
			}
		case "dst==up":
			gate := append([]float32(nil), gateBase...)
			up := append([]float32(nil), upBase...)
			if !SiLUMulExpTo(up, gate, up, scratch) {
				t.Fatalf("%s returned false", mode)
			}
			for i := range up {
				if !siluMulApproxEqual(up[i], want[i]) {
					t.Fatalf("%s[%d]=%g want %g", mode, i, up[i], want[i])
				}
			}
		case "dst==gate==up":
			all := append([]float32(nil), gateBase...)
			if !SiLUMulExpTo(all, all, all, scratch) {
				t.Fatalf("%s returned false", mode)
			}
			for i := range all {
				if !siluMulApproxEqual(all[i], want[i]) {
					t.Fatalf("%s[%d]=%g want %g", mode, i, all[i], want[i])
				}
			}
		}
	}
}

func TestSiLUMulExpToRandomAccuracy(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	gate := make([]float32, 50000)
	up := make([]float32, len(gate))
	for i := range gate {
		gate[i] = rng.Float32()*200 - 100
		up[i] = rng.Float32()*200 - 100
	}
	dst := make([]float32, len(gate))
	scratch := make([]float32, len(gate))
	if !SiLUMulExpTo(dst, gate, up, scratch) {
		t.Fatal("SiLUMulExpTo returned false for random input")
	}
	for i := range dst {
		want := siluMulReference(gate[i], up[i])
		if !siluMulApproxEqual(dst[i], want) {
			t.Fatalf("random[%d] gate=%g up=%g got=%g want=%g", i, gate[i], up[i], dst[i], want)
		}
	}
}

func TestSiLUMulExpToExceptionalClassification(t *testing.T) {
	gate := []float32{
		float32(math.NaN()),
		float32(math.NaN()),
		float32(math.Inf(1)),
		float32(math.Inf(1)),
		float32(math.Inf(-1)),
		float32(math.Inf(-1)),
		float32(math.Copysign(0, -1)),
		0,
		-150,
		-150,
		120,
		120,
	}
	up := []float32{1, 0, 0, -2, 1, -3, 4, -5, 1, -2, 0, 2}
	dst := make([]float32, len(gate))
	scratch := make([]float32, len(gate))
	if !SiLUMulExpTo(dst, gate, up, scratch) {
		t.Fatal("SiLUMulExpTo returned false for exceptional input")
	}
	for i := range dst {
		want := siluMulReference(gate[i], up[i])
		if math.IsNaN(float64(want)) {
			if !math.IsNaN(float64(dst[i])) {
				t.Fatalf("exceptional[%d] got=%v want NaN", i, dst[i])
			}
			continue
		}
		if math.Float32bits(dst[i]) != math.Float32bits(want) {
			t.Fatalf("exceptional[%d] got bits=0x%08x want bits=0x%08x got=%v want=%v", i, math.Float32bits(dst[i]), math.Float32bits(want), dst[i], want)
		}
	}
}

func TestSiLUMulExpToTails(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	for _, n := range []int{1, 2, 3, 7, 8, 9, 15, 16, 17, 31} {
		gateBacking := make([]float32, n+2)
		upBacking := make([]float32, n+2)
		dstBacking := make([]float32, n+2)
		scratchBacking := make([]float32, n+2)
		gateBacking[0], gateBacking[n+1] = 111, 222
		upBacking[0], upBacking[n+1] = 333, 444
		dstBacking[0], dstBacking[n+1] = 555, 666
		scratchBacking[0], scratchBacking[n+1] = 777, 888
		gate := gateBacking[1 : n+1]
		up := upBacking[1 : n+1]
		dst := dstBacking[1 : n+1]
		scratch := scratchBacking[1 : n+1]
		for i := 0; i < n; i++ {
			gate[i] = rng.Float32()*20 - 10
			up[i] = rng.Float32()*20 - 10
			dst[i] = -999
			scratch[i] = -999
		}
		if !SiLUMulExpTo(dst, gate, up, scratch) {
			t.Fatalf("n=%d returned false", n)
		}
		for i := 0; i < n; i++ {
			want := siluMulReference(gate[i], up[i])
			if !siluMulApproxEqual(dst[i], want) {
				t.Fatalf("n=%d i=%d got=%g want=%g", n, i, dst[i], want)
			}
		}
		if gateBacking[0] != 111 || gateBacking[n+1] != 222 ||
			upBacking[0] != 333 || upBacking[n+1] != 444 ||
			dstBacking[0] != 555 || dstBacking[n+1] != 666 ||
			scratchBacking[0] != 777 || scratchBacking[n+1] != 888 {
			t.Fatalf("n=%d mutated tails gate=%v up=%v dst=%v scratch=%v", n, gateBacking, upBacking, dstBacking, scratchBacking)
		}
	}
}

func TestSiLUMulExpToNoAllocsFFN128x3072(t *testing.T) {
	const n = 128 * 3072
	gate := make([]float32, n)
	up := make([]float32, n)
	dst := make([]float32, n)
	scratch := make([]float32, n)
	for i := range gate {
		gate[i] = float32(i%257-128) * 0.5
		up[i] = float32(i%131-65) * 0.25
	}
	if allocs := testing.AllocsPerRun(10, func() {
		if !SiLUMulExpTo(dst, gate, up, scratch) {
			panic("SiLUMulExpTo returned false")
		}
	}); allocs != 0 {
		t.Fatalf("allocs %g", allocs)
	}
}

func BenchmarkSiLUMulExpToFFN128x3072(b *testing.B) {
	const n = 128 * 3072
	gate := make([]float32, n)
	up := make([]float32, n)
	dst := make([]float32, n)
	scratch := make([]float32, n)
	for i := range gate {
		gate[i] = float32(i%257-128) * 0.5
		up[i] = float32(i%131-65) * 0.25
	}
	bench := func(b *testing.B, fn func()) {
		b.ReportAllocs()
		b.SetBytes(int64(n * 4 * 4))
		for range b.N {
			fn()
		}
	}
	b.Run("baseline_SiLUMulTo", func(b *testing.B) {
		bench(b, func() {
			if !SiLUMulTo(dst, gate, up) {
				b.Fatal("SiLUMulTo returned false")
			}
		})
	})
	b.Run("exp_scratch", func(b *testing.B) {
		bench(b, func() {
			if !SiLUMulExpTo(dst, gate, up, scratch) {
				b.Fatal("SiLUMulExpTo returned false")
			}
		})
	})
}
