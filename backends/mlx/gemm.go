package mlx

import (
	"runtime"
	"sync"
)

// Gemm performs a small batched MLX quantized matrix-vector multiply.
//
// x is [batch, inDim], out is [batch, outDim], both row-major. The current
// implementation reuses the validated scalar GEMV path per row; it provides a
// stable API for future prefill-oriented AVX2/NEON kernels without changing
// callers.
func Gemm(out, x []float32, batch int, qw *QuantWeight) bool {
	if batch <= 0 || ValidateQuantWeight(qw) != nil {
		return false
	}
	xLen, okX := checkedMulInt(batch, qw.InDim)
	outLen, okOut := checkedMulInt(batch, qw.OutDim)
	if !okX || !okOut || len(x) < xLen || len(out) < outLen {
		return false
	}
	if batch < 2 || qw.OutDim*qw.InDim < 4096 {
		for b := 0; b < batch; b++ {
			xRow := x[b*qw.InDim : (b+1)*qw.InDim]
			outRow := out[b*qw.OutDim : (b+1)*qw.OutDim]
			GemvTo(outRow, xRow, qw)
		}
		return true
	}

	workers := runtime.GOMAXPROCS(0)
	if workers > batch {
		workers = batch
	}
	chunk := (batch + workers - 1) / workers
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		start := w * chunk
		end := start + chunk
		if end > batch {
			end = batch
		}
		if start >= end {
			continue
		}
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			for b := start; b < end; b++ {
				xRow := x[b*qw.InDim : (b+1)*qw.InDim]
				outRow := out[b*qw.OutDim : (b+1)*qw.OutDim]
				GemvTo(outRow, xRow, qw)
			}
		}(start, end)
	}
	wg.Wait()
	return true
}
