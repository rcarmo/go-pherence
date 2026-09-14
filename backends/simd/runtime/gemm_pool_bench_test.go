package simd

import (
	"context"
	"fmt"
	"runtime"
	"testing"
)

func BenchmarkGEMMPool(b *testing.B) {
	old := runtime.GOMAXPROCS(4)
	defer runtime.GOMAXPROCS(old)

	const (
		m = 128
		n = 1024
		k = 1024
	)
	a := make([]float32, m*k)
	w := make([]float32, n*k)
	c := make([]float32, m*n)
	for i := range a {
		a[i] = float32(i%31-15) / 32
	}
	for i := range w {
		w[i] = float32(i%29-14) / 32
	}
	packed, err := PackSgemmNTWeights(w, n, k, k)
	if err != nil {
		b.Fatal(err)
	}
	for _, workers := range []int{1, 2, 4} {
		for _, tc := range []struct {
			name   string
			packed []float32
		}{
			{name: "streamed"},
			{name: "prepacked", packed: packed},
		} {
			b.Run(fmt.Sprintf("w%d/%s", workers, tc.name), func(b *testing.B) {
				pool, err := NewGEMMPool(workers, k)
				if err != nil {
					b.Fatal(err)
				}
				defer pool.Close()
				if err := pool.Run(context.Background(), c, a, w, tc.packed, m, n, k, 1, k, k, n); err != nil {
					b.Fatal(err)
				}
				clear(c)
				b.ReportAllocs()
				b.ResetTimer()
				for b.Loop() {
					clear(c)
					if err := pool.Run(context.Background(), c, a, w, tc.packed, m, n, k, 1, k, k, n); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}
