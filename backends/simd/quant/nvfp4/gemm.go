package nvfp4

import (
	"runtime"
	"sync"
)

// GemmNVFP4 computes out[batch,outDim] = x[batch,inDim] @ W^T directly from
// packed NVFP4 weights. It is the scalar/reference batched counterpart to
// GemvNVFP4To and provides a stable interface for future prefill/native paths.
func GemmNVFP4(out, x []float32, batch int, qw *NVFP4Weight) bool {
	if batch <= 0 || ValidateNVFP4Weight(qw) != nil {
		return false
	}
	xLen, okX := checkedMulInt(batch, qw.InDim)
	outLen, okOut := checkedMulInt(batch, qw.OutDim)
	if !okX || !okOut || len(x) < xLen || len(out) < outLen {
		return false
	}
	if RuntimeCapabilities().HasGemv {
		return gemmNVFP4Accelerated(out[:outLen], x[:xLen], batch, qw)
	}
	if batch == 1 {
		return GemvNVFP4To(out[:qw.OutDim], x[:qw.InDim], qw)
	}
	workers := runtime.GOMAXPROCS(0)
	if batch < 2 || qw.OutDim*qw.InDim < 4096 || workers <= 1 {
		for b := 0; b < batch; b++ {
			if !GemvNVFP4To(out[b*qw.OutDim:(b+1)*qw.OutDim], x[b*qw.InDim:(b+1)*qw.InDim], qw) {
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
				if !GemvNVFP4To(out[b*qw.OutDim:(b+1)*qw.OutDim], x[b*qw.InDim:(b+1)*qw.InDim], qw) {
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
