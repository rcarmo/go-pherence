package simd

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func TestSoftmaxSIMDParity(t *testing.T) {
	r := rand.New(rand.NewSource(4))
	for _, n := range []int{1, 7, 8, 9, 31, 128, 1025} {
		for trial := 0; trial < 20; trial++ {
			a := make([]float32, n)
			for i := range a {
				a[i] = float32(r.NormFloat64() * 10)
			}
			b := append([]float32(nil), a...)
			if !SoftmaxSIMDInPlace(a) || !SoftmaxInPlace(b) {
				t.Fatal("softmax failed")
			}
			var sum float64
			for i := range a {
				if math.Abs(float64(a[i]-b[i])) > 2e-6 {
					t.Fatalf("n=%d difference %g", n, a[i]-b[i])
				}
				sum += float64(a[i])
			}
			if math.Abs(sum-1) > 3e-6 {
				t.Fatal("sum", sum)
			}
		}
	}
	for _, input := range [][]float32{nil, {float32(math.Inf(1)), 0}, {float32(math.NaN()), 1}, {float32(math.Inf(-1)), float32(math.Inf(-1))}} {
		if SoftmaxSIMDInPlace(input) {
			t.Fatal("accepted invalid input")
		}
	}
	x := make([]float32, 128)
	if n := testing.AllocsPerRun(10, func() { SoftmaxSIMDInPlace(x) }); n != 0 {
		t.Fatal(n)
	}
}

func BenchmarkSoftmaxSIMD(b *testing.B) {
	for _, n := range []int{128, 218} {
		for _, fast := range []bool{false, true} {
			name := "scalar"
			if fast {
				name = "simd"
			}
			b.Run(fmt.Sprintf("%s/%d", name, n), func(b *testing.B) {
				x := make([]float32, n)
				orig := make([]float32, n)
				for i := range orig {
					orig[i] = float32(i%17 - 8)
				}
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					copy(x, orig)
					if fast {
						SoftmaxSIMDInPlace(x)
					} else {
						SoftmaxInPlace(x)
					}
				}
			})
		}
	}
}

func TestSoftmaxSIMDBoundaries(t *testing.T) {
	for _, row := range [][]float32{{1000, 999, 968, 0}, {0, -32, -32.0001, -80}, {7, 7, 7, 7, 7, 7, 7, 7}, {0, -100, -100, -100, -100, -100, -100, -100, -100}, {0, 0, 0, 0, -32, -32, -32, -32, -32.0001}} {
		want := append([]float32(nil), row...)
		if !SoftmaxInPlace(want) || !SoftmaxSIMDInPlace(row) {
			t.Fatal("failed")
		}
		for i := range row {
			if math.Abs(float64(row[i]-want[i])) > 2e-6 {
				t.Fatalf("%v != %v", row, want)
			}
		}
	}
}
