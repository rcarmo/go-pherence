package k3

import (
	"testing"

	"github.com/rcarmo/go-pherence/backends/spacemit/k3/aipool"
)

func BenchmarkQ4KMatVec4PooledVsFourMatVecs(b *testing.B) {
	M, K := 1024, 1024
	w := makeBenchQ4X32(M, K)
	var acts [4][]float32
	var outs [4][]float32
	pool := aipool.NewAIWorkerPool(6)
	defer pool.Close()
	for r := 0; r < 4; r++ {
		acts[r] = make([]float32, K)
		outs[r] = make([]float32, M)
		for k := range acts[r] {
			acts[r][k] = float32(((r+3)*(k%37) - 17)) / 11.0
		}
	}
	b.Run("four_independent_matvecs", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			for r := 0; r < 4; r++ {
				q4kQ41x32MatVecGoAsm(w, acts[r], outs[r], pool)
			}
		}
	})
	b.Run("pooled_m4_dispatch", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			if !q4kQ41x32MatVec4Pooled(w, acts, outs, pool) {
				b.Fatal("pooled m4 failed")
			}
		}
	})
}

func TestQ4KMatVec4PooledMatchesFourM1(t *testing.T) {
	t.Skip("k3I8I4M4 Go fallback has format mismatch; M4 path not used in single-token decode")
}
