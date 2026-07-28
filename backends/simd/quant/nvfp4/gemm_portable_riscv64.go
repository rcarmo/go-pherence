//go:build riscv64

package nvfp4

import (
	"runtime"
	"sync"
)

func gemmNVFP4Portable(out, x []float32, batch int, qw *NVFP4Weight) bool {
	workers := runtime.GOMAXPROCS(0)
	if qw.OutDim < 4096 || workers <= 1 {
		for b := 0; b < batch; b++ {
			gemvNVFP4Rows(out[b*qw.OutDim:(b+1)*qw.OutDim], x[b*qw.InDim:(b+1)*qw.InDim], qw, 0, qw.OutDim)
		}
		return true
	}
	if workers > qw.OutDim {
		workers = qw.OutDim
	}
	rowsPer := (qw.OutDim + workers - 1) / workers
	var wg sync.WaitGroup
	for start := 0; start < qw.OutDim; start += rowsPer {
		end := start + rowsPer
		if end > qw.OutDim {
			end = qw.OutDim
		}
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			for b := 0; b < batch; b++ {
				gemvNVFP4Rows(out[b*qw.OutDim:(b+1)*qw.OutDim], x[b*qw.InDim:(b+1)*qw.InDim], qw, start, end)
			}
		}(start, end)
	}
	wg.Wait()
	return true
}
