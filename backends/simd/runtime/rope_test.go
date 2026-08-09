package simd

import (
	"math"
	"math/rand"
	"testing"

	"golang.org/x/sys/cpu"
)

func TestApplyRoPEPartialAcceleratedExact(t *testing.T) {
	rng := rand.New(rand.NewSource(0x20fe2026))
	for heads := 1; heads <= 9; heads++ {
		for headDim := 2; headDim <= 130; headDim += 2 {
			for _, rotHalf := range []int{1, headDim / 4, headDim / 2} {
				if rotHalf < 1 {
					continue
				}
				const pos = 2
				x := make([]float32, heads*headDim)
				freqs := make([]float32, (pos+1)*rotHalf*2)
				for i := range x {
					x[i] = rng.Float32()*20 - 10
				}
				for i := range freqs {
					freqs[i] = rng.Float32()*2 - 1
				}
				want, got := append([]float32(nil), x...), append([]float32(nil), x...)
				applyRoPEPartialGo(want, freqs, pos, heads, headDim, rotHalf)
				ApplyRoPEPartial(got, freqs, pos, heads, headDim, rotHalf)
				for i := range got {
					if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
						t.Fatalf("heads=%d headDim=%d rotHalf=%d i=%d got=%08x want=%08x", heads, headDim, rotHalf, i, math.Float32bits(got[i]), math.Float32bits(want[i]))
					}
				}
			}
		}
	}
}

func BenchmarkApplyRoPEPartial(b *testing.B) {
	const heads, headDim, rotHalf, pos = 16, 256, 128, 2
	x := make([]float32, heads*headDim)
	freqs := make([]float32, (pos+1)*rotHalf*2)
	for i := range x {
		x[i] = float32(i%31-15) / 7
	}
	for i := range freqs {
		freqs[i] = float32(i%17-8) / 9
	}
	b.Run("plan9-avx2", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			ApplyRoPEPartial(x, freqs, pos, heads, headDim, rotHalf)
		}
	})
	b.Run("retained-scalar", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			applyRoPEPartialGo(x, freqs, pos, heads, headDim, rotHalf)
		}
	})
}

func TestApplyRoPEPartialUnsupportedCPUFallback(t *testing.T) {
	x := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	freqs := []float32{0, 1, 1, 0, -1, 0, 0, -1}
	want, got := append([]float32(nil), x...), append([]float32(nil), x...)
	applyRoPEPartialGo(want, freqs, 0, 2, 4, 2)
	old := cpu.X86.HasAVX2
	cpu.X86.HasAVX2 = false
	defer func() { cpu.X86.HasAVX2 = old }()
	ApplyRoPEPartial(got, freqs, 0, 2, 4, 2)
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("i=%d got=%08x want=%08x", i, math.Float32bits(got[i]), math.Float32bits(want[i]))
		}
	}
}

func TestApplyRoPECheckedRuntimeWrappers(t *testing.T) {
	x := []float32{1, 2, 3, 4, 99}
	if !ApplyRoPEPartialTo(x, []float32{0, 1}, 0, 1, 4, 1) {
		t.Fatal("ApplyRoPEPartialTo returned false for valid input")
	}
	want := []float32{-2, 1, 3, 4, 99}
	for i := range want {
		if x[i] != want[i] {
			t.Fatalf("x[%d]=%g want %g (all=%v)", i, x[i], want[i], x)
		}
	}
	if ApplyRoPEPartialTo(x, []float32{0}, 0, 1, 4, 1) {
		t.Fatal("ApplyRoPEPartialTo accepted short freqs")
	}
	if ApplyRoPEPartialTo(x[:3], []float32{0, 1}, 0, 1, 4, 1) {
		t.Fatal("ApplyRoPEPartialTo accepted short x")
	}
	if ApplyRoPETo(x[:4], []float32{0, 1, 1, 0}, 0, 1, 4) == false {
		t.Fatal("ApplyRoPETo rejected valid full RoPE")
	}
}

func TestApplyRoPEPartialRuntimeWrapper(t *testing.T) {
	x := []float32{1, 2, 3, 4, 99}
	ApplyRoPEPartial(x, []float32{0, 1}, 0, 1, 4, 1)
	want := []float32{-2, 1, 3, 4, 99}
	for i := range want {
		if x[i] != want[i] {
			t.Fatalf("x[%d]=%g want %g (all=%v)", i, x[i], want[i], x)
		}
	}
}
