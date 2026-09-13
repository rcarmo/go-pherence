package simd

import (
	"math"
	"math/rand"
	"testing"
)

func TestSiLUFinishBitwiseAndAliases(t *testing.T) {
	rng := rand.New(rand.NewSource(72))
	for _, n := range []int{1, 7, 8, 9, 15, 16, 17, 256, 1025} {
		for _, mode := range []string{"separate", "gate", "up", "both"} {
			gate, up, ex, want := make([]float32, n), make([]float32, n), make([]float32, n), make([]float32, n)
			for i := range gate {
				gate[i] = float32(rng.NormFloat64() * 10)
				up[i] = float32(rng.NormFloat64() * 4)
				ex[i] = float32(math.Exp(float64(-gate[i])))
			}
			if mode == "both" {
				up = gate
			}
			for i := range want {
				want[i] = (gate[i] / (1 + ex[i])) * up[i]
			}
			dst := make([]float32, n)
			if mode == "gate" || mode == "both" {
				dst = gate
			}
			if mode == "up" {
				dst = up
			}
			siluFinish(dst, gate, up, ex)
			for i, v := range dst {
				if math.Float32bits(v) != math.Float32bits(want[i]) {
					t.Fatalf("n%d %s i%d got%08x want%08x", n, mode, i, math.Float32bits(v), math.Float32bits(want[i]))
				}
			}
		}
	}
}

func TestSiLUFinishExceptionalLanes(t *testing.T) {
	values := []float32{0, math.Float32frombits(0x80000000), 1, -1, math.SmallestNonzeroFloat32, math.MaxFloat32, float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN())}
	for _, v := range values {
		gate, up, ex, dst := make([]float32, 16), make([]float32, 16), make([]float32, 16), make([]float32, 16)
		for i := range gate {
			gate[i] = v
			up[i] = float32(i - 8)
			ex[i] = float32(math.Exp(float64(-v)))
		}
		siluFinish(dst, gate, up, ex)
		for i, x := range dst {
			want := (gate[i] / (1 + ex[i])) * up[i]
			if math.IsNaN(float64(want)) {
				if !math.IsNaN(float64(x)) {
					t.Fatal("NaN lost")
				}
			} else if math.Float32bits(x) != math.Float32bits(want) {
				t.Fatalf("%g lane%d got%08x want%08x", v, i, math.Float32bits(x), math.Float32bits(want))
			}
		}
	}
}

func TestSiLUScratchOverlapNoMutation(t *testing.T) {
	for _, which := range []string{"up-exact", "gate-partial", "dst-partial"} {
		backing := make([]float32, 17)
		for i := range backing {
			backing[i] = float32(i + 1)
		}
		dst, gate, up := make([]float32, 16), make([]float32, 16), make([]float32, 16)
		scratch := backing[1:]
		switch which {
		case "up-exact":
			up = scratch
		case "gate-partial":
			gate = backing[:16]
		case "dst-partial":
			dst = backing[:16]
		}
		before := append([]float32(nil), backing...)
		if SiLUMulExpTo(dst, gate, up, scratch) {
			t.Fatalf("accepted %s", which)
		}
		for i := range backing {
			if backing[i] != before[i] {
				t.Fatalf("mutated %s", which)
			}
		}
	}
}
