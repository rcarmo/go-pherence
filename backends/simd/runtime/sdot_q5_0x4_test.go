package simd

import (
	"math"
	"math/rand"
	"testing"
)

// Every row of the four-row kernel must be bit-identical to SdotQ5_0.
func TestSdotQ5_0x4MatchesSingleRow(t *testing.T) {
	r := rand.New(rand.NewSource(11))
	for _, n := range []int{32, 64, 96, 1280, 5120} {
		for _, pad := range []int{0, 8, 33} {
			stride := n + pad
			for trial := 0; trial < 100; trial++ {
				raw := randomQ5Row(r, n/32)
				x := make([]float32, 3*stride+n)
				for i := range x {
					switch r.Intn(50) {
					case 0:
						x[i] = 0
					case 1:
						x[i] = math.Float32frombits(r.Uint32() & 0x807fffff)
					case 2:
						x[i] = float32(math.Inf(1 - 2*r.Intn(2)))
					default:
						x[i] = float32(r.NormFloat64() * math.Pow(10, float64(r.Intn(7)-3)))
					}
				}
				var got [4]float32
				SdotQ5_0x4(x, stride, raw, &got)
				for row := 0; row < 4; row++ {
					want := SdotQ5_0(x[row*stride:row*stride+n], raw)
					if math.Float32bits(got[row]) != math.Float32bits(want) && !(math.IsNaN(float64(got[row])) && math.IsNaN(float64(want))) {
						t.Fatalf("n=%d pad=%d trial=%d row=%d got %08x want %08x", n, pad, trial, row, math.Float32bits(got[row]), math.Float32bits(want))
					}
				}
			}
		}
	}
	if !HasSdotQ5_0Asm {
		t.Log("fused kernel not admitted on this CPU; reference path checked")
	}
}

func TestSdotQ5_0x4RejectsShape(t *testing.T) {
	for _, c := range []struct{ x, stride, raw int }{{4 * 32, 32, 21}, {3*32 + 31, 32, 22}, {4 * 64, 31, 22}} {
		func() {
			defer func() {
				if recover() == nil {
					t.Fatal("accepted", c)
				}
			}()
			var out [4]float32
			SdotQ5_0x4(make([]float32, c.x), c.stride, make([]byte, c.raw), &out)
		}()
	}
}

func BenchmarkSdotQ5_0x4Row1280(b *testing.B) {
	r := rand.New(rand.NewSource(1))
	raw := randomQ5Row(r, 40)
	x := make([]float32, 4*1280)
	for i := range x {
		x[i] = float32(r.NormFloat64())
	}
	var out [4]float32
	b.Run("x4", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			SdotQ5_0x4(x, 1280, raw, &out)
		}
	})
	b.Run("4xsingle", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			for row := 0; row < 4; row++ {
				out[row] = SdotQ5_0(x[row*1280:(row+1)*1280], raw)
			}
		}
	})
}
