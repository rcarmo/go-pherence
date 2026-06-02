package main

import (
	"math"
	"testing"
)

func TestQ4KMatVec4PooledMatchesFourM1(t *testing.T) {
	M, K := 1024, 1024
	w := makeBenchQ4X32(M, K)
	if !w.Valid {
		t.Fatal("repack failed")
	}
	var acts [4][]float32
	var got [4][]float32
	var want [4][]float32
	pool := NewAIWorkerPool(6)
	defer pool.Close()
	for r := 0; r < 4; r++ {
		acts[r] = make([]float32, K)
		got[r] = make([]float32, M)
		want[r] = make([]float32, M)
		for k := range acts[r] {
			acts[r][k] = float32(((r+3)*(k%37)-17)) / 11.0
		}
		q4kQ41x32MatVecGoAsm(w, acts[r], want[r], pool)
	}
	if !q4kQ41x32MatVec4Pooled(w, acts, got, pool) {
		t.Fatal("q4kQ41x32MatVec4Pooled returned false")
	}
	var maxDiff float64
	maxR, maxI := -1, -1
	for r := 0; r < 4; r++ {
		for i := 0; i < M; i++ {
			if d := math.Abs(float64(got[r][i] - want[r][i])); d > maxDiff {
				maxDiff, maxR, maxI = d, r, i
			}
		}
	}
	t.Logf("maxDiff=%.6f row=%d idx=%d got=%.6f want=%.6f", maxDiff, maxR, maxI, got[maxR][maxI], want[maxR][maxI])
	if maxDiff > 0.06 {
		t.Fatalf("maxDiff %.6f > tolerance", maxDiff)
	}
}

func BenchmarkQ4KMatVec4PooledVsFourMatVecs(b *testing.B) {
	M, K := 1024, 1024
	w := makeBenchQ4X32(M, K)
	var acts [4][]float32
	var outs [4][]float32
	pool := NewAIWorkerPool(6)
	defer pool.Close()
	for r := 0; r < 4; r++ {
		acts[r] = make([]float32, K)
		outs[r] = make([]float32, M)
		for k := range acts[r] {
			acts[r][k] = float32(((r+3)*(k%37)-17)) / 11.0
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
