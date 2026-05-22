package q4

import (
	"runtime"
	"sync"
)

// Gemm computes out[batch,outDim] = x[batch,inDim] @ W^T for GPTQ INT4
// weights. It supports symmetric and asymmetric qzeros layouts and provides a
// scalar/reference batched API for prefill and future AVX2/NEON kernels.
func Gemm(out, x []float32, batch int, qweight, qzeros, gIdx []int32, scales []float32, inDim, outDim int, sym bool) bool {
	if batch <= 0 {
		return false
	}
	xLen, okX := checkedMulInt(batch, inDim)
	outLen, okOut := checkedMulInt(batch, outDim)
	if !okX || !okOut || len(x) < xLen || len(out) < outLen {
		return false
	}
	if err := Validate(qweight, qzeros, gIdx, scales, inDim, outDim, sym); err != nil {
		return false
	}
	caps := RuntimeCapabilities()
	if sym && caps.HasGemvSym {
		return gemmSymAccelerated(out[:outLen], x[:xLen], batch, qweight, gIdx, scales, inDim, outDim)
	}
	if !sym && caps.HasDequant {
		return gemmAccelerated(out[:outLen], x[:xLen], batch, qweight, qzeros, gIdx, scales, inDim, outDim)
	}
	if batch == 1 {
		return GemvTo(out[:outDim], x[:inDim], qweight, qzeros, gIdx, scales, inDim, outDim, sym)
	}
	workers := runtime.GOMAXPROCS(0)
	if batch < 2 || inDim*outDim < 4096 || workers <= 1 {
		for b := 0; b < batch; b++ {
			if !GemvTo(out[b*outDim:(b+1)*outDim], x[b*inDim:(b+1)*inDim], qweight, qzeros, gIdx, scales, inDim, outDim, sym) {
				return false
			}
		}
		return true
	}
	if workers > batch {
		workers = batch
	}
	chunk := (batch + workers - 1) / workers
	ok := true
	var mu sync.Mutex
	var wg sync.WaitGroup
	for start := 0; start < batch; start += chunk {
		end := start + chunk
		if end > batch {
			end = batch
		}
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			for b := start; b < end; b++ {
				if !GemvTo(out[b*outDim:(b+1)*outDim], x[b*inDim:(b+1)*inDim], qweight, qzeros, gIdx, scales, inDim, outDim, sym) {
					mu.Lock()
					ok = false
					mu.Unlock()
					return
				}
			}
		}(start, end)
	}
	wg.Wait()
	return ok
}

// GemmSym computes batched symmetric GPTQ INT4 GEMM.
func GemmSym(out, x []float32, batch int, qweight, gIdx []int32, scales []float32, inDim, outDim int) bool {
	return Gemm(out, x, batch, qweight, nil, gIdx, scales, inDim, outDim, true)
}
