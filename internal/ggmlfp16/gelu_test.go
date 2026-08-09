package ggmlfp16

import (
	"math"
	"math/rand"
	"testing"

	"github.com/rcarmo/go-pherence/half"
)

func TestGELUFP16LookupMulToExact(t *testing.T) {
	gate := make([]float32, 0, (1<<16)+1007)
	for i := 0; i < 1<<16; i++ {
		gate = append(gate, half.F16ToF32(uint16(i)))
	}
	gate = append(gate, -10, math.Nextafter32(-10, 0), 10, math.Nextafter32(10, 0), 0, float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN()))
	rng := rand.New(rand.NewSource(0x6e1f16))
	for range 999 {
		gate = append(gate, math.Float32frombits(rng.Uint32()))
	}
	up := make([]float32, len(gate))
	want := make([]float32, len(gate))
	for i := range up {
		up[i] = math.Float32frombits(rng.Uint32())
		want[i] = GELUFP16Lookup(gate[i]) * up[i]
	}
	got := append([]float32(nil), gate...)
	if !GELUFP16LookupMulTo(got, got, up) {
		t.Fatal("valid buffers rejected")
	}
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("i=%d gate=%08x up=%08x got=%08x want=%08x", i, math.Float32bits(gate[i]), math.Float32bits(up[i]), math.Float32bits(got[i]), math.Float32bits(want[i]))
		}
	}
}

func BenchmarkGELUFP16LookupMulTo(b *testing.B) {
	const n = 124 * 10240
	gate, up, dst := make([]float32, n), make([]float32, n), make([]float32, n)
	rng := rand.New(rand.NewSource(0x6e1f16))
	for i := range gate {
		gate[i], up[i] = (rng.Float32()*2-1)*12, rng.Float32()*2-1
	}
	b.Run("plan9-avx2", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			GELUFP16LookupMulTo(dst, gate, up)
		}
	})
	b.Run("retained-scalar", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			for j := range dst {
				dst[j] = GELUFP16Lookup(gate[j]) * up[j]
			}
		}
	})
}

func TestGELUFP16LookupMulToUnsupportedCPUFallback(t *testing.T) {
	gate := []float32{-11, -1, 0, 1, 11, 0.5, -0.5, 3, 4}
	up := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}
	want := make([]float32, len(gate))
	for i := range want {
		want[i] = GELUFP16Lookup(gate[i]) * up[i]
	}
	old := hasF16C
	hasF16C = false
	defer func() { hasF16C = old }()
	if !GELUFP16LookupMulTo(gate, gate, up) {
		t.Fatal("valid buffers rejected")
	}
	for i := range gate {
		if math.Float32bits(gate[i]) != math.Float32bits(want[i]) {
			t.Fatalf("i=%d got=%08x want=%08x", i, math.Float32bits(gate[i]), math.Float32bits(want[i]))
		}
	}
}

func TestGELUFP16LookupMulToTails(t *testing.T) {
	for n := 0; n < 24; n++ {
		gate, up, want := make([]float32, n), make([]float32, n), make([]float32, n)
		for i := range gate {
			gate[i], up[i] = float32(i-11)/3, float32(i+1)/7
			want[i] = GELUFP16Lookup(gate[i]) * up[i]
		}
		if !GELUFP16LookupMulTo(gate, gate, up) {
			t.Fatalf("n=%d rejected", n)
		}
		for i := range gate {
			if math.Float32bits(gate[i]) != math.Float32bits(want[i]) {
				t.Fatalf("n=%d i=%d got=%08x want=%08x", n, i, math.Float32bits(gate[i]), math.Float32bits(want[i]))
			}
		}
	}
}
