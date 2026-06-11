package aipool

import (
	"os"
	"runtime"
	"strconv"
	"sync"

	"github.com/rcarmo/go-pherence/backends/spacemit/ime2"
)

func q8ActPackWorkers(groups int) int {
	if groups <= 1 {
		return 1
	}
	n := runtime.GOMAXPROCS(0)
	if v := os.Getenv("IME2_ACT_PACK_WORKERS"); v != "" {
		if x, err := strconv.Atoi(v); err == nil && x > 0 {
			n = x
		}
	}
	if n < 1 {
		n = 1
	}
	if n > groups {
		n = groups
	}
	return n
}

func packQ80M4ActivationsX100(x []float32, M, K, kBlks int, gelu bool) ([]byte, int) {
	groups := (M + 3) / 4
	stride := kBlks * ime2.K3I8I8ABlockM4Bytes
	aData := make([]byte, groups*stride)
	workers := q8ActPackWorkers(groups)
	zeroRow := make([]float32, K)
	var wg sync.WaitGroup
	wg.Add(workers)
	for wid := 0; wid < workers; wid++ {
		wid := wid
		go func() {
			defer wg.Done()
			g0 := wid * groups / workers
			g1 := (wid + 1) * groups / workers
			for g := g0; g < g1; g++ {
				r := g * 4
				actual := 4
				if M-r < actual {
					actual = M - r
				}
				var rows [4][]float32
				for i := 0; i < 4; i++ {
					if i < actual {
						rows[i] = x[(r+i)*K : (r+i+1)*K]
					} else {
						rows[i] = zeroRow
					}
				}
				dst := aData[g*stride : (g+1)*stride]
				if gelu {
					ime2.QuantizeF32RowsQ8M4GELUInto(rows, kBlks, dst)
				} else {
					ime2.QuantizeF32RowsQ8M4Into(rows, kBlks, dst)
				}
			}
		}()
	}
	wg.Wait()
	return aData, groups
}

func gemmQ80x32AIPooledPackedA(aData []byte, groups, M, kBlks int, w ime2.Q80x32, out []float32, pool *AIWorkerPool) bool {
	n := w.M
	stride := kBlks * ime2.K3I8I8ABlockM4Bytes
	pool.Run(func(workerID, nWorkers int) {
		g0 := workerID * groups / nWorkers
		g1 := (workerID + 1) * groups / nWorkers
		if g1 <= g0 {
			return
		}
		tailOut := make([]float32, 4*n)
		for g := g0; g < g1; g++ {
			r := g * 4
			actual := 4
			if M-r < actual {
				actual = M - r
			}
			a := aData[g*stride : (g+1)*stride]
			if actual == 4 {
				ime2.K3I8I8(&a[0], &w.BData[0], &out[r*n], 4, n, kBlks, n)
				continue
			}
			for i := range tailOut {
				tailOut[i] = 0
			}
			ime2.K3I8I8(&a[0], &w.BData[0], &tailOut[0], 4, n, kBlks, n)
			for i := 0; i < actual; i++ {
				copy(out[(r+i)*n:(r+i+1)*n], tailOut[i*n:(i+1)*n])
			}
		}
	})
	return true
}

// GemmQ80x32AIPooledX100Pack computes the same result as GemmQ80x32AIPooled,
// but quantizes/pack activations on normal X100 goroutines before dispatching
// packed A blocks to the A100 worker pool. It is intended to measure whether
// scalar activation packing on A100 workers is the limiting factor.
func GemmQ80x32AIPooledX100Pack(x []float32, M, K int, w ime2.Q80x32, out []float32, pool *AIWorkerPool) bool {
	if pool == nil || !w.Valid || w.K != K || K%32 != 0 || w.M%32 != 0 || M <= 0 || len(out) < M*w.M || len(x) < M*K {
		return false
	}
	kBlks := K / 32
	aData, groups := packQ80M4ActivationsX100(x, M, K, kBlks, false)
	return gemmQ80x32AIPooledPackedA(aData, groups, M, kBlks, w, out, pool)
}

// Gemm2Q80x32AIPooledX100PackSameInput computes two row-scale Q8 GEMMs with
// the same activation matrix. Activations are packed once on X100 goroutines and
// each A100 worker dispatch applies both B matrices to its row-group range. This
// is intended for gated MLPs (W1/W3) that share the same input.
func Gemm2Q80x32AIPooledX100PackSameInput(x []float32, M, K int, wA, wB ime2.Q80x32, outA, outB []float32, pool *AIWorkerPool) bool {
	if pool == nil || !wA.Valid || !wB.Valid || wA.K != K || wB.K != K || wA.M != wB.M || K%32 != 0 || wA.M%32 != 0 || M <= 0 || len(outA) < M*wA.M || len(outB) < M*wB.M || len(x) < M*K {
		return false
	}
	kBlks := K / 32
	aData, groups := packQ80M4ActivationsX100(x, M, K, kBlks, false)
	n := wA.M
	stride := kBlks * ime2.K3I8I8ABlockM4Bytes
	pool.Run(func(workerID, nWorkers int) {
		g0 := workerID * groups / nWorkers
		g1 := (workerID + 1) * groups / nWorkers
		if g1 <= g0 {
			return
		}
		tailA := make([]float32, 4*n)
		tailB := make([]float32, 4*n)
		for g := g0; g < g1; g++ {
			r := g * 4
			actual := 4
			if M-r < actual {
				actual = M - r
			}
			a := aData[g*stride : (g+1)*stride]
			if actual == 4 {
				ime2.K3I8I8(&a[0], &wA.BData[0], &outA[r*n], 4, n, kBlks, n)
				ime2.K3I8I8(&a[0], &wB.BData[0], &outB[r*n], 4, n, kBlks, n)
				continue
			}
			for i := range tailA {
				tailA[i] = 0
				tailB[i] = 0
			}
			ime2.K3I8I8(&a[0], &wA.BData[0], &tailA[0], 4, n, kBlks, n)
			ime2.K3I8I8(&a[0], &wB.BData[0], &tailB[0], 4, n, kBlks, n)
			for i := 0; i < actual; i++ {
				copy(outA[(r+i)*n:(r+i+1)*n], tailA[i*n:(i+1)*n])
				copy(outB[(r+i)*n:(r+i+1)*n], tailB[i*n:(i+1)*n])
			}
		}
	})
	return true
}

// GemmQ80x32AIPooledGELUX100Pack is the X100 activation-pack variant for the
// fused GELU+FC2 path.
func GemmQ80x32AIPooledGELUX100Pack(x []float32, M, K int, w ime2.Q80x32, out []float32, pool *AIWorkerPool) bool {
	if pool == nil || !w.Valid || w.K != K || K%32 != 0 || w.M%32 != 0 || M <= 0 || len(out) < M*w.M || len(x) < M*K {
		return false
	}
	kBlks := K / 32
	aData, groups := packQ80M4ActivationsX100(x, M, K, kBlks, true)
	return gemmQ80x32AIPooledPackedA(aData, groups, M, kBlks, w, out, pool)
}

func packQ80M4ActivationsX100GELURowScale(x []float32, M, K, kBlks int) ([]byte, int) {
	groups := (M + 3) / 4
	stride := kBlks * ime2.K3I8I8ABlockM4Bytes
	aData := make([]byte, groups*stride)
	workers := q8ActPackWorkers(groups)
	zeroRow := make([]float32, K)
	var wg sync.WaitGroup
	wg.Add(workers)
	for wid := 0; wid < workers; wid++ {
		wid := wid
		go func() {
			defer wg.Done()
			g0 := wid * groups / workers
			g1 := (wid + 1) * groups / workers
			for g := g0; g < g1; g++ {
				r := g * 4
				actual := 4
				if M-r < actual {
					actual = M - r
				}
				var rows [4][]float32
				for i := 0; i < 4; i++ {
					if i < actual {
						rows[i] = x[(r+i)*K : (r+i+1)*K]
					} else {
						rows[i] = zeroRow
					}
				}
				dst := aData[g*stride : (g+1)*stride]
				ime2.QuantizeF32RowsQ8M4GELURowScaleInto(rows, kBlks, dst)
			}
		}()
	}
	wg.Wait()
	return aData, groups
}

// GemmQ80x32AIPooledGELUX100PackRowScale is the native-int8-compatible X100
// activation-pack variant for the fused GELU+FC2 path.
func GemmQ80x32AIPooledGELUX100PackRowScale(x []float32, M, K int, w ime2.Q80x32, out []float32, pool *AIWorkerPool) bool {
	if pool == nil || !w.Valid || w.K != K || K%32 != 0 || w.M%32 != 0 || M <= 0 || len(out) < M*w.M || len(x) < M*K {
		return false
	}
	kBlks := K / 32
	aData, groups := packQ80M4ActivationsX100GELURowScale(x, M, K, kBlks)
	return gemmQ80x32AIPooledPackedA(aData, groups, M, kBlks, w, out, pool)
}
