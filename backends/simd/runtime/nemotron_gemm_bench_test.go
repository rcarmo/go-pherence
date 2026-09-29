package simd

import (
	"context"
	"testing"
)

// Diagnostic for the bounded Nemotron diarisation tower shapes. Weight
// packing is outside the timed region, as in a loaded model.
func BenchmarkNemotronBoundedNTPaths(b *testing.B) {
	for _, shape := range []struct {
		label   string
		m, n, k int
	}{
		{"asr-qkv-5", 5, 1024, 1024}, {"asr-ff1-5", 5, 4096, 1024}, {"asr-ff2-5", 5, 1024, 4096},
		{"qkv-320", 320, 512, 512}, {"qkv-541", 541, 512, 512},
		{"mlp1-320", 320, 2048, 512}, {"mlp1-541", 541, 2048, 512},
		{"mlp2-320", 320, 512, 2048}, {"mlp2-541", 541, 512, 2048},
		{"scores-320", 320, 320, 64}, {"scores-541", 541, 541, 64},
	} {
		a := randFloats(shape.m*shape.k, 11)
		weights := randFloats(shape.n*shape.k, 17)
		packed, err := PackSgemmNTWeights(weights, shape.n, shape.k, shape.k)
		if err != nil {
			b.Fatal(err)
		}
		out := make([]float32, shape.m*shape.n)
		b.Run(shape.label+"/blocked", func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				clear(out)
				if !DenseNTTo(out, a, weights, shape.m, shape.n, shape.k, 1, shape.k, shape.k, shape.n) {
					b.Fatal("blocked rejected")
				}
			}
		})
		b.Run(shape.label+"/prepacked", func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				clear(out)
				if !SgemmNTPrepackedTo(out, a, weights, packed, shape.m, shape.n, shape.k, 1, shape.k, shape.k, shape.n) {
					b.Fatal("prepacked rejected")
				}
			}
		})
		for _, workers := range []int{2, 4} {
			b.Run(shape.label+"/pool-columns-w"+string(rune('0'+workers)), func(b *testing.B) {
				pool, err := NewGEMMPool(workers, shape.k)
				if err != nil {
					b.Fatal(err)
				}
				defer pool.Close()
				ctx := context.Background()
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					clear(out)
					if err := pool.RunColumns(ctx, out, a, weights, packed, shape.m, shape.n, shape.k, 1, shape.k, shape.k, shape.n); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}
