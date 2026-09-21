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

	shapes := []struct {
		m, n, k int
	}{
		{m: 75, n: 1024, k: 1024},
		{m: 75, n: 3072, k: 1024},
		{m: 210, n: 1024, k: 1024},
		{m: 210, n: 3072, k: 1024},
	}
	runners := []struct {
		name string
		run  func(*GEMMPool, context.Context, []float32, []float32, []float32, []float32, int, int, int, float32, int, int, int) error
	}{
		{name: "rows", run: (*GEMMPool).Run},
		{name: "columns", run: (*GEMMPool).RunColumns},
	}
	for _, shape := range shapes {
		b.Run(fmt.Sprintf("m%d/n%d/k%d", shape.m, shape.n, shape.k), func(b *testing.B) {
			a := make([]float32, shape.m*shape.k)
			w := make([]float32, shape.n*shape.k)
			c := make([]float32, shape.m*shape.n)
			for i := range a {
				a[i] = float32(i%31-15) / 32
			}
			for i := range w {
				w[i] = float32(i%29-14) / 32
			}
			packed, err := PackSgemmNTWeights(w, shape.n, shape.k, shape.k)
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
					for _, runner := range runners {
						b.Run(fmt.Sprintf("w%d/%s/%s", workers, tc.name, runner.name), func(b *testing.B) {
							pool, err := NewGEMMPool(workers, shape.k)
							if err != nil {
								b.Fatal(err)
							}
							defer pool.Close()
							if err := runner.run(pool, context.Background(), c, a, w, tc.packed, shape.m, shape.n, shape.k, 1, shape.k, shape.k, shape.n); err != nil {
								b.Fatal(err)
							}
							clear(c)
							b.ReportAllocs()
							b.ResetTimer()
							for b.Loop() {
								clear(c)
								if err := runner.run(pool, context.Background(), c, a, w, tc.packed, shape.m, shape.n, shape.k, 1, shape.k, shape.k, shape.n); err != nil {
									b.Fatal(err)
								}
							}
						})
					}
				}
			}
		})
	}
}
