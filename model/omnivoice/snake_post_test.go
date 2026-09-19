package omnivoice

import (
	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"math"
	"testing"
)

func legacySnakeChannel(row []float32, alpha float32) {
	var storage [256]float32
	scale := float32(1) / (alpha + 1e-9)
	for start := 0; start < len(row); start += len(storage) {
		x := row[start:min(start+len(storage), len(row))]
		tmp := storage[:len(x)]
		simd.VecScale(tmp, x, alpha)
		simd.SinF32To(tmp, tmp)
		snakePostVectors(x, tmp, scale)
	}
}
func TestSnakePostExact(t *testing.T) {
	for _, n := range []int{0, 1, 7, 8, 9, 15, 16, 17, 255, 256, 257, 777, 4096} {
		for _, alpha := range []float32{0, 1e-8, .01, .5, 1, 4, 32, -.5} {
			want, got := make([]float32, n), make([]float32, n)
			for i := range want {
				want[i] = float32(math.Sin(float64(i)*.31) * 34)
				got[i] = want[i]
			}
			legacySnakeChannel(want, alpha)
			snakeChannel(got, alpha)
			for i := range want {
				if math.Float32bits(want[i]) != math.Float32bits(got[i]) {
					t.Fatalf("n%d alpha%g index%d", n, alpha, i)
				}
			}
		}
	}
}
func TestSnakePostExceptionalAndBounds(t *testing.T) {
	vals := []float32{0, float32(math.Copysign(0, -1)), 1, -1, math.SmallestNonzeroFloat32, math.MaxFloat32, float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN())}
	for n := 0; n <= 33; n++ {
		for _, scale := range vals {
			want, got, sine, tmp := make([]float32, n+2), make([]float32, n+2), make([]float32, n), make([]float32, n)
			for i := range want {
				want[i] = 123
				got[i] = 123
			}
			for i := range sine {
				sine[i] = vals[i%len(vals)]
				tmp[i] = sine[i]
				want[i+1] = vals[(i+3)%len(vals)]
				got[i+1] = want[i+1]
			}
			snakePostVectors(want[1:n+1], tmp, scale)
			snakePost(got[1:n+1], sine, scale)
			for i := range got {
				if math.IsNaN(float64(want[i])) {
					if !math.IsNaN(float64(got[i])) {
						t.Fatal("NaN classification")
					}
				} else if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
					t.Fatalf("n%d scale%g index%d: %g vs %g", n, scale, i, got[i], want[i])
				}
			}
		}
	}
}
