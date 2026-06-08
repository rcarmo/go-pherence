package aipool

import "github.com/rcarmo/go-pherence/backends/spacemit/ime2"

// GemmQ80x32AIPooled computes out[M,N] = x[M,K] · w[N,K]^T using the native
// A100 Q8_0 x Q8_0 N32/K32 K3I8I8 kernels. w must come from
// ime2.PackF32ToQ80x32(N,K,weights). M is processed in groups of four rows;
// tail rows are currently ignored and should be handled by the caller/fallback.
func GemmQ80x32AIPooled(x []float32, M, K int, w ime2.Q80x32, out []float32, pool *AIWorkerPool) bool {
	if pool == nil || !w.Valid || w.K != K || K%32 != 0 || w.M%32 != 0 || M < 4 || len(out) < M*w.M || len(x) < M*K {
		return false
	}
	kBlks := K / 32
	n := w.M
	m4 := (M / 4) * 4
	pool.Run(func(workerID, nWorkers int) {
		row0 := (workerID * m4 / nWorkers / 4) * 4
		row1 := ((workerID + 1) * m4 / nWorkers / 4) * 4
		if workerID == nWorkers-1 {
			row1 = m4
		}
		if row1 <= row0 {
			return
		}
		aScratch := make([]byte, kBlks*ime2.K3I8I8ABlockM4Bytes)
		for r := row0; r < row1; r += 4 {
			var rows [4][]float32
			for i := 0; i < 4; i++ {
				rows[i] = x[(r+i)*K : (r+i+1)*K]
			}
			a := ime2.QuantizeF32RowsQ8M4Into(rows, kBlks, aScratch)
			ime2.K3I8I8(&a[0], &w.BData[0], &out[r*n], 4, n, kBlks, n)
		}
	})
	return true
}
