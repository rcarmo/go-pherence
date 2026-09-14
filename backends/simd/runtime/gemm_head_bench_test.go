package simd

import (
	"context"
	"fmt"
	"runtime"
	"testing"
)

// Actual chunk width of OmniVoice audio heads; rows include sparse late steps.
func BenchmarkGEMMAudioHeadDispatch(b *testing.B) {
	old := runtime.GOMAXPROCS(2)
	defer runtime.GOMAXPROCS(old)
	for _, m := range []int{6, 12, 24, 48, 78, 126} {
		for _, mode := range []string{"serial", "rows", "columns"} {
			b.Run(fmt.Sprintf("m%d/%s", m, mode), func(b *testing.B) {
				const n, k = 128, 1024
				a, w, c := make([]float32, m*k), make([]float32, n*k), make([]float32, m*n)
				for i := range a {
					a[i] = float32(i%23-11) / 23
				}
				for i := range w {
					w[i] = float32(i%31-15) / 31
				}
				scratch := make([]float32, k*16)
				pool, err := NewGEMMPool(2, k)
				if err != nil {
					b.Fatal(err)
				}
				defer pool.Close()
				run := func() {
					clear(c)
					switch mode {
					case "serial":
						if !SgemmNTPackedTo(c, a, w, scratch, m, n, k, 1, k, k, n) {
							b.Fatal("invalid")
						}
					case "rows":
						if err := pool.Run(context.Background(), c, a, w, nil, m, n, k, 1, k, k, n); err != nil {
							b.Fatal(err)
						}
					case "columns":
						if err := pool.RunColumns(context.Background(), c, a, w, nil, m, n, k, 1, k, k, n); err != nil {
							b.Fatal(err)
						}
					}
				}
				run()
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					run()
				}
			})
		}
	}
}
