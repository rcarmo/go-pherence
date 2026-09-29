package simd

import (
	"runtime"
	"sync"

	"github.com/rcarmo/go-pherence/internal/checked"
)

// SgemmNTPrepackedParallelTo partitions complete 16-column panels across at
// most four workers. Each worker calls the checked prepacked kernel on
// disjoint output columns, so each output accumulates in the same order as
// SgemmNTPrepackedTo. For small jobs and unavailable SIMD it uses that kernel
// directly. The caller owns all slices; no goroutine survives this call.
func SgemmNTPrepackedParallelTo(c, a, weights, packed []float32, m, n, k int, alpha float32, lda, ldb, ldc int) bool {
	if !validSgemmSliceArgs(c, a, weights, m, n, k, lda, ldb, ldc, true) {
		return false
	}
	panels, packedLen, ok := checkedSgemmNTFullPanelLayout(n, k)
	if !ok || len(packed) < packedLen || packedLen > 0 && (!float32SlicesDisjoint(packed[:packedLen], c) || !float32SlicesDisjoint(packed[:packedLen], a) || !float32SlicesDisjoint(packed[:packedLen], weights)) {
		return false
	}
	workers := min(runtime.GOMAXPROCS(0), 4, panels)
	// Preserve the serial API's behaviour for overlapping raw operands;
	// partitioning such input would introduce concurrent reads and writes.
	if !HasSgemmAsm || m < 64 || n < 256 || k < 256 || workers < 2 || !float32SlicesDisjoint(c, a) || !float32SlicesDisjoint(c, weights) {
		return SgemmNTPrepackedTo(c, a, weights, packed, m, n, k, alpha, lda, ldb, ldc)
	}
	// Check partition arithmetic before writing to C. The packed panel
	// footprint was checked above; n, ldb and ldc came through slice checks.
	panelStride, ok := checked.MulInt(k, gebpNR)
	if !ok {
		return false
	}
	var wg sync.WaitGroup
	results := make([]bool, workers)
	for worker := 0; worker < workers; worker++ {
		start := worker * panels / workers
		end := (worker + 1) * panels / workers
		col := start * gebpNR
		cols := (end - start) * gebpNR
		wg.Add(1)
		go func(index, col, cols, start int) {
			defer wg.Done()
			results[index] = SgemmNTPrepackedTo(c[col:], a, weights[col*ldb:], packed[start*panelStride:], m, cols, k, alpha, lda, ldb, ldc)
		}(worker, col, cols, start)
	}
	wg.Wait()
	for _, result := range results {
		if !result {
			return false
		}
	}
	if tail := panels * gebpNR; tail < n {
		return SgemmNTTo(c[tail:], a, weights[tail*ldb:], m, n-tail, k, alpha, lda, ldb, ldc)
	}
	return true
}
