package simd

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"
)

func randomQ5Row(r *rand.Rand, blocks int) []byte {
	raw := make([]byte, blocks*Q5_0BlockBytes)
	scales := []uint16{0x0001, 0x03ff, 0x0400, 0x1c00, 0x3c00, 0xbc00, 0x2e66, 0xa1a3, 0x0000, 0x8000, 0x7bff, 0xfbff}
	for b := 0; b < blocks; b++ {
		block := raw[b*Q5_0BlockBytes:][:Q5_0BlockBytes]
		var bits uint16
		if r.Intn(4) == 0 {
			bits = scales[r.Intn(len(scales))]
		} else {
			for bits = uint16(r.Uint32()); bits&0x7c00 == 0x7c00; bits = uint16(r.Uint32()) {
			}
		}
		binary.LittleEndian.PutUint16(block, bits)
		r.Read(block[2:])
	}
	return raw
}

// The fused kernel must be bit-identical to Sdot over the exactly widened row.
func TestSdotQ5_0MatchesWidenedSdot(t *testing.T) {
	r := rand.New(rand.NewSource(5))
	for _, n := range []int{32, 64, 96, 1280, 5120} {
		for trial := 0; trial < 200; trial++ {
			raw := randomQ5Row(r, n/32)
			x := make([]float32, n)
			for i := range x {
				switch r.Intn(50) {
				case 0:
					x[i] = 0
				case 1:
					x[i] = math.Float32frombits(r.Uint32() & 0x807fffff) // subnormal
				default:
					x[i] = float32(r.NormFloat64() * math.Pow(10, float64(r.Intn(7)-3)))
				}
			}
			w := make([]float32, n)
			DequantQ5_0Into(w, raw)
			want := Sdot(x, w)
			got := SdotQ5_0(x, raw)
			if math.Float32bits(got) != math.Float32bits(want) && !(math.IsNaN(float64(got)) && math.IsNaN(float64(want))) {
				t.Fatalf("n=%d trial=%d got %v (%08x) want %v (%08x)", n, trial, got, math.Float32bits(got), want, math.Float32bits(want))
			}
		}
	}
	if !HasSdotQ5_0Asm {
		t.Log("fused kernel not admitted on this CPU; reference path checked")
	}
}

func TestDequantQ5_0IntoLayout(t *testing.T) {
	raw := make([]byte, Q5_0BlockBytes)
	binary.LittleEndian.PutUint16(raw, 0x3c00) // d = 1
	binary.LittleEndian.PutUint32(raw[2:], 0x80000001)
	raw[6] = 0x21  // q0 low=1, q16 high=2
	raw[21] = 0xf3 // q15 low=3, q31 high=15
	w := make([]float32, 32)
	DequantQ5_0Into(w, raw)
	if w[0] != 1+16-16 || w[16] != 2-16 || w[15] != 3-16 || w[31] != 15+16-16 || w[1] != -16 {
		t.Fatal(w)
	}
}

func TestSdotQ5_0RejectsShape(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("accepted short raw")
		}
	}()
	SdotQ5_0(make([]float32, 32), make([]byte, 21))
}

func BenchmarkSdotQ5_0Row5120(b *testing.B) {
	r := rand.New(rand.NewSource(1))
	raw := randomQ5Row(r, 160)
	x := make([]float32, 5120)
	w := make([]float32, 5120)
	DequantQ5_0Into(w, raw)
	for i := range x {
		x[i] = float32(r.NormFloat64())
	}
	b.Run("fused", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			SdotQ5_0(x, raw)
		}
	})
	b.Run("f32", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			Sdot(x, w)
		}
	})
}
