package simd

import (
	"context"
	"fmt"
	"math"
	"testing"
)

// BenchmarkGEMMColumnHotset keeps the caller's GOMAXPROCS setting. Use
// GOMAXPROCS=2 for the two-vCPU OmniVoice configuration. Both raw and packed
// weights stay live in each mode, matching the resident-prepacked API contract.
// Matrices rotate between calls to distinguish a repeatedly reused projection
// from a larger working set. Setup, packing and exact parity checks are untimed.
func BenchmarkGEMMColumnHotset(b *testing.B) {
	for _, shape := range []struct{ m, n, k int }{
		{126, 1024, 1024},
		{126, 3072, 1024},
		{128, 1024, 3072},
	} {
		for _, count := range []int{1, 8} {
			b.Run(fmt.Sprintf("m%d/n%d/k%d/set%d", shape.m, shape.n, shape.k, count), func(b *testing.B) {
				a := make([]float32, shape.m*shape.k)
				for i := range a {
					a[i] = float32(i%31-15) / 32
				}
				weights := make([][]float32, count)
				packed := make([][]float32, count)
				for j := range weights {
					weights[j] = make([]float32, shape.n*shape.k)
					for i := range weights[j] {
						weights[j][i] = float32((i+j*7)%29-14) / 32
					}
					var err error
					packed[j], err = PackSgemmNTWeights(weights[j], shape.n, shape.k, shape.k)
					if err != nil {
						b.Fatal(err)
					}
				}
				pool, err := NewGEMMPool(2, shape.k)
				if err != nil {
					b.Fatal(err)
				}
				defer pool.Close()
				ctx := context.Background()
				c := make([]float32, shape.m*shape.n)
				want := make([]float32, len(c))
				for j := range weights {
					clear(c)
					clear(want)
					if err := pool.RunColumns(ctx, want, a, weights[j], nil, shape.m, shape.n, shape.k, 1, shape.k, shape.k, shape.n); err != nil {
						b.Fatal(err)
					}
					if err := pool.RunColumns(ctx, c, a, weights[j], packed[j], shape.m, shape.n, shape.k, 1, shape.k, shape.k, shape.n); err != nil {
						b.Fatal(err)
					}
					for i := range c {
						if math.Float32bits(c[i]) != math.Float32bits(want[i]) {
							b.Fatalf("matrix %d element %d: raw %g, packed %g", j, i, want[i], c[i])
						}
					}
				}
				for _, mode := range []string{"raw", "prepacked"} {
					b.Run(mode, func(b *testing.B) {
						b.ReportAllocs()
						b.ResetTimer()
						j := 0
						for b.Loop() {
							clear(c)
							var p []float32
							if mode == "prepacked" {
								p = packed[j]
							}
							if err := pool.RunColumns(ctx, c, a, weights[j], p, shape.m, shape.n, shape.k, 1, shape.k, shape.k, shape.n); err != nil {
								b.Fatal(err)
							}
							j++
							if j == count {
								j = 0
							}
						}
					})
				}
			})
		}
	}
}
