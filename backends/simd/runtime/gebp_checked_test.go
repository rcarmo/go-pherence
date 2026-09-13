package simd

import (
	"math"
	"testing"
)

func TestPackedGEMMShapes(t *testing.T) {
	for _, shape := range [][3]int{{1, 1, 1}, {5, 17, 7}, {6, 16, 32}, {7, 33, 9}, {13, 32, 67}, {128, 128, 256}} {
		m, n, k := shape[0], shape[1], shape[2]
		a, b, c := make([]float32, m*k), make([]float32, n*k), make([]float32, m*n)
		want := make([]float32, len(c))
		for i := range a {
			a[i] = float32(i%11-5) * .03
		}
		for i := range b {
			b[i] = float32(i%17-8) * .02
		}
		for i := range c {
			c[i] = .1
			want[i] = .1
		}
		for r := 0; r < m; r++ {
			for col := 0; col < n; col++ {
				sum := float32(0)
				for j := 0; j < k; j++ {
					sum += a[r*k+j] * b[col*k+j]
				}
				want[r*n+col] += .7 * sum
			}
		}
		scratch := make([]float32, k*16)
		if !SgemmNTPackedTo(c, a, b, scratch, m, n, k, .7, k, k, n) {
			t.Fatal(shape)
		}
		for i, v := range c {
			if diff := math.Abs(float64(v - want[i])); diff > 1e-4 || math.IsNaN(diff) {
				t.Fatalf("%v at %d diff %g", shape, i, diff)
			}
		}
		if allocations := testing.AllocsPerRun(5, func() { clear(c); SgemmNTPackedTo(c, a, b, scratch, m, n, k, 1, k, k, n) }); allocations != 0 {
			t.Fatalf("allocs %g", allocations)
		}
	}
}
func TestPackedGEMMRejectsShort(t *testing.T) {
	if SgemmNTPackedTo(make([]float32, 16), make([]float32, 16), make([]float32, 16), nil, 4, 4, 4, 1, 4, 4, 4) {
		t.Fatal("scratch accepted")
	}
	if SgemmNTPackedTo(nil, nil, nil, nil, 1, 1, 1, 1, 1, 1, 1) {
		t.Fatal("short buffers accepted")
	}
}
