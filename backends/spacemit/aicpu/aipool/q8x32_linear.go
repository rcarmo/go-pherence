package aipool

import "github.com/rcarmo/go-pherence/backends/spacemit/ime2"

// GemmQ80x32AIPooled computes out[M,N] = x[M,K] · w[N,K]^T using the native
// A100 Q8_0 x Q8_0 N32/K32 K3I8I8 kernels. w must come from
// ime2.PackF32ToQ80x32(N,K,weights). Non-M4 tails are handled internally with a
// zero-row/tail-output scratch so callers do not need to clone/pad the full
// activation matrix.
func GemmQ80x32AIPooled(x []float32, M, K int, w ime2.Q80x32, out []float32, pool *AIWorkerPool) bool {
	if pool == nil || !w.Valid || w.K != K || K%32 != 0 || w.M%32 != 0 || M <= 0 || len(out) < M*w.M || len(x) < M*K {
		return false
	}
	kBlks := K / 32
	n := w.M
	groups := (M + 3) / 4
	pool.Run(func(workerID, nWorkers int) {
		g0 := workerID * groups / nWorkers
		g1 := (workerID + 1) * groups / nWorkers
		if g1 <= g0 {
			return
		}
		aScratch := make([]byte, kBlks*ime2.K3I8I8ABlockM4Bytes)
		zeroRow := make([]float32, K)
		tailOut := make([]float32, 4*n)
		for g := g0; g < g1; g++ {
			r := g * 4
			var rows [4][]float32
			actual := 4
			if M-r < actual {
				actual = M - r
			}
			for i := 0; i < 4; i++ {
				if i < actual {
					rows[i] = x[(r+i)*K : (r+i+1)*K]
				} else {
					rows[i] = zeroRow
				}
			}
			a := ime2.QuantizeF32RowsQ8M4Into(rows, kBlks, aScratch)
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

// GemmQ80x32AIPooledGELU is GemmQ80x32AIPooled with GELU fused into activation
// quantization. It computes out[M,N] = GELU(x[M,K]) · w[N,K]^T. This avoids a
// separate GELU pass and a second full hidden write before A100 FC2.
func GemmQ80x32AIPooledGELU(x []float32, M, K int, w ime2.Q80x32, out []float32, pool *AIWorkerPool) bool {
	if pool == nil || !w.Valid || w.K != K || K%32 != 0 || w.M%32 != 0 || M <= 0 || len(out) < M*w.M || len(x) < M*K {
		return false
	}
	kBlks := K / 32
	n := w.M
	groups := (M + 3) / 4
	pool.Run(func(workerID, nWorkers int) {
		g0 := workerID * groups / nWorkers
		g1 := (workerID + 1) * groups / nWorkers
		if g1 <= g0 {
			return
		}
		aScratch := make([]byte, kBlks*ime2.K3I8I8ABlockM4Bytes)
		zeroRow := make([]float32, K)
		tailOut := make([]float32, 4*n)
		for g := g0; g < g1; g++ {
			r := g * 4
			var rows [4][]float32
			actual := 4
			if M-r < actual {
				actual = M - r
			}
			for i := 0; i < 4; i++ {
				if i < actual {
					rows[i] = x[(r+i)*K : (r+i+1)*K]
				} else {
					rows[i] = zeroRow
				}
			}
			a := ime2.QuantizeF32RowsQ8M4GELUInto(rows, kBlks, aScratch)
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
