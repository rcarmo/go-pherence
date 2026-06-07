package k3engine

import (
	"runtime"
	"testing"
	"unsafe"

	"github.com/rcarmo/go-pherence/backends/spacemit/ime2"
	"github.com/rcarmo/go-pherence/backends/spacemit/k3engine/aipool"
)

func makeBenchQ8X32(M, K int) q8Q80x32 {
	f32 := make([]float32, M*K)
	for m := 0; m < M; m++ {
		for k := 0; k < K; k++ {
			f32[m*K+k] = float32(((m*11+k*3)%47)-23) / 17.0
		}
	}
	return repackF32ToQ80x32(M, K, f32)
}

func TestQ8I8MatVec4M4MatchesFourM1(t *testing.T) {
	aipool.RegisterAIThread(8)
	M, K := 64, 1024
	subs := K / 32
	w := makeBenchQ8X32(M, K)
	if !w.Valid {
		t.Fatal("repack failed")
	}
	var acts [4][]float32
	var outs [4][]float32
	var want [4][]float32
	var qRows [4][]byte
	for r := 0; r < 4; r++ {
		acts[r] = make([]float32, K)
		outs[r] = make([]float32, M)
		want[r] = make([]float32, M)
		for k := range acts[r] {
			acts[r][k] = float32(((r+1)*(k%31) - 15)) / 10.0
		}
		qRows[r] = quantizeQ8Blocks32Bytes(acts[r])
		for rg := 0; rg < M/32; rg++ {
			k3I8I8M1((*byte)(unsafe.Pointer(&qRows[r][0])), (*byte)(unsafe.Pointer(&w.BData[rg*subs*1088])), &want[r][rg*32], subs, 32)
		}
	}
	if !q8Q80x32MatVec4Native(w, acts, outs) {
		t.Fatal("q8Q80x32MatVec4Native returned false")
	}
	var maxDiff float64
	maxR, maxI := -1, -1
	for r := 0; r < 4; r++ {
		for i := 0; i < M; i++ {
			if d := abs64(float64(outs[r][i] - want[r][i])); d > maxDiff {
				maxDiff, maxR, maxI = d, r, i
			}
		}
	}
	t.Logf("maxDiff=%.6f row=%d idx=%d got=%.6f want=%.6f", maxDiff, maxR, maxI, outs[maxR][maxI], want[maxR][maxI])
	if maxDiff > 0.05 {
		t.Fatalf("maxDiff %.6f > tolerance", maxDiff)
	}
}

func BenchmarkK3I8I8M1x4VsM4(b *testing.B) {
	K := 1024
	subs := K / 32
	w := makeBenchQ8X32(32, K)
	var acts [4][]float32
	var qRows [4][]byte
	for r := 0; r < 4; r++ {
		acts[r] = make([]float32, K)
		for k := range acts[r] {
			acts[r][k] = float32(((r+1)*(k%31) - 15)) / 10.0
		}
		qRows[r] = quantizeQ8Blocks32Bytes(acts[r])
	}
	packedA := ime2.PackQ8RowsM4(qRows, subs)
	outM1 := make([]float32, 4*32)
	outM4 := make([]float32, 4*32)
	b.Run("four_m1", func(b *testing.B) {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		aipool.RegisterAIThread(8)
		for i := 0; i < b.N; i++ {
			for r := 0; r < 4; r++ {
				k3I8I8M1((*byte)(unsafe.Pointer(&qRows[r][0])), (*byte)(unsafe.Pointer(&w.BData[0])), &outM1[r*32], subs, 32)
			}
		}
	})
	b.Run("m4_dispatch", func(b *testing.B) {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		aipool.RegisterAIThread(8)
		for i := 0; i < b.N; i++ {
			handled := q8I8Dispatcher((*byte)(unsafe.Pointer(&packedA[0])), (*byte)(unsafe.Pointer(&w.BData[0])), &outM4[0], 4, 32, subs, 32)
			if handled != 4 {
				b.Fatalf("handled=%d", handled)
			}
		}
	})
}

func abs64(v float64) float64 {
	if v < 0 {
		return -v
	}
	return v
}

func TestQ8I8MatVec4PooledMatchesFourM1(t *testing.T) {
	M, K := 1024, 1024
	w := makeBenchQ8X32(M, K)
	if !w.Valid {
		t.Fatal("repack failed")
	}
	var acts [4][]float32
	var got [4][]float32
	var want [4][]float32
	pool := aipool.NewAIWorkerPool(6)
	defer pool.Close()
	for r := 0; r < 4; r++ {
		acts[r] = make([]float32, K)
		got[r] = make([]float32, M)
		want[r] = make([]float32, M)
		for k := range acts[r] {
			acts[r][k] = float32(((r+5)*(k%29) - 13)) / 12.0
		}
		q8Q80x32MatVecNative(w, acts[r], want[r], pool)
	}
	if !q8Q80x32MatVec4Pooled(w, acts, got, pool) {
		t.Fatal("q8Q80x32MatVec4Pooled returned false")
	}
	var maxDiff float64
	maxR, maxI := -1, -1
	for r := 0; r < 4; r++ {
		for i := 0; i < M; i++ {
			if d := abs64(float64(got[r][i] - want[r][i])); d > maxDiff {
				maxDiff, maxR, maxI = d, r, i
			}
		}
	}
	t.Logf("maxDiff=%.6f row=%d idx=%d got=%.6f want=%.6f", maxDiff, maxR, maxI, got[maxR][maxI], want[maxR][maxI])
	if maxDiff > 0.06 {
		t.Fatalf("maxDiff %.6f > tolerance", maxDiff)
	}
}

func BenchmarkQ8I8MatVec4PooledVsFourMatVecs(b *testing.B) {
	M, K := 1024, 1024
	w := makeBenchQ8X32(M, K)
	var acts [4][]float32
	var outs [4][]float32
	pool := aipool.NewAIWorkerPool(6)
	defer pool.Close()
	for r := 0; r < 4; r++ {
		acts[r] = make([]float32, K)
		outs[r] = make([]float32, M)
		for k := range acts[r] {
			acts[r][k] = float32(((r+5)*(k%29) - 13)) / 12.0
		}
	}
	b.Run("four_independent_matvecs", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			for r := 0; r < 4; r++ {
				q8Q80x32MatVecNative(w, acts[r], outs[r], pool)
			}
		}
	})
	b.Run("pooled_m4_dispatch", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			if !q8Q80x32MatVec4Pooled(w, acts, outs, pool) {
				b.Fatal("pooled m4 failed")
			}
		}
	})
}

func TestQ8I8NativeBWaveMatVecMatchesDirect(t *testing.T) {
	old := aipool.Int8TCMBWaveOn
	defer func() { aipool.Int8TCMBWaveOn = old }()
	M, K := 1024, 1024
	w := makeBenchQ8X32(M, K)
	if !w.Valid {
		t.Fatal("repack failed")
	}
	act := make([]float32, K)
	for k := range act {
		act[k] = float32((k%37)-18) / 9.0
	}
	pool := aipool.NewAIWorkerPool(6)
	defer pool.Close()
	direct := make([]float32, M)
	wave := make([]float32, M)
	aipool.Int8TCMBWaveOn = false
	if !q8Q80x32MatVecNative(w, act, direct, pool) {
		t.Fatal("direct native matvec failed")
	}
	aipool.Int8TCMBWaveOn = true
	if !q8Q80x32MatVecNative(w, act, wave, pool) {
		t.Fatal("b-wave native matvec failed")
	}
	var maxDiff float64
	maxIdx := -1
	for i := range direct {
		if d := abs64(float64(direct[i] - wave[i])); d > maxDiff {
			maxDiff, maxIdx = d, i
		}
	}
	if maxIdx >= 0 {
		t.Logf("maxDiff=%.6f idx=%d direct=%.6f wave=%.6f", maxDiff, maxIdx, direct[maxIdx], wave[maxIdx])
	} else {
		t.Logf("maxDiff=0")
	}
	if maxDiff > 1e-5 {
		t.Fatalf("maxDiff %.6f > tolerance", maxDiff)
	}
}
