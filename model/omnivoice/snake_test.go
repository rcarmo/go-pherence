package omnivoice

import (
	"math"
	"testing"
)

func scalarSnakeChannel(x []float32, a float32) {
	for i, v := range x {
		phase := a * v
		s := float32(math.Sin(float64(phase)))
		x[i] += (float32(1) / (a + 1e-9)) * (s * s)
	}
}
func TestSnakeChannelParityAndAllocation(t *testing.T) {
	for _, n := range []int{1, 7, 8, 15, 256, 257, 1024} {
		for _, a := range []float32{0, 1e-8, 0.01, 0.5, 1, 4, 32} {
			x, want := make([]float32, n), make([]float32, n)
			for i := range x {
				x[i] = float32(i%137-68) / 8
			}
			copy(want, x)
			scalarSnakeChannel(want, a)
			snakeChannel(x, a)
			for i := range x {
				if diff := math.Abs(float64(x[i] - want[i])); diff > 2e-5 || math.IsNaN(diff) {
					t.Fatalf("n%d a%g i%d got%g want%g", n, a, i, x[i], want[i])
				}
			}
		}
	}
	row := make([]float32, 1024)
	if n := testing.AllocsPerRun(20, func() { snakeChannel(row, 1) }); n != 0 {
		t.Fatalf("allocations %g", n)
	}
}
func BenchmarkSnakeChannel(b *testing.B) {
	for _, fast := range []bool{false, true} {
		name := "scalar"
		if fast {
			name = "simd"
		}
		b.Run(name, func(b *testing.B) {
			x := make([]float32, 4096)
			src := make([]float32, len(x))
			for i := range src {
				src[i] = float32(i%137-68) / 8
			}
			b.ReportAllocs()
			for b.Loop() {
				copy(x, src)
				if fast {
					snakeChannel(x, 0.7)
				} else {
					scalarSnakeChannel(x, 0.7)
				}
			}
		})
	}
}

func TestSnakeChannelDispatchBoundaries(t *testing.T) {
	src := make([]float32, 777)
	for i := range src {
		src[i] = float32(i%11-5) / 8
	}
	values := []float32{math.Nextafter32(32, 0), 32, math.Nextafter32(32, float32(math.Inf(1))), -math.Nextafter32(32, 0), -32, -math.Nextafter32(32, float32(math.Inf(1)))}
	for _, pos := range []int{1, 250, 254, 255, 256, 257, 509, 513, 770} {
		copy(src[pos:], values)
	}
	want := append([]float32(nil), src...)
	scalarSnakeChannel(want, 1)
	snakeChannel(src, 1)
	for i, v := range src {
		if diff := math.Abs(float64(v) - float64(want[i])); diff > 2e-5 || math.IsNaN(diff) {
			t.Fatalf("index%d got%g want%g", i, v, want[i])
		}
	}
}
