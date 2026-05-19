package nvfp4

import (
	"runtime"
	"sync"
)

// GemvNVFP4 performs a correctness-first matrix-vector multiply directly from
// packed NVFP4 weights: out[outDim] = W_nvfp4[outDim, inDim] · x[inDim].
func GemvNVFP4(out, x []float32, qw *NVFP4Weight) {
	if err := ValidateNVFP4Weight(qw); err != nil || len(out) < qw.OutDim || len(x) < qw.InDim {
		return
	}
	workers := runtime.GOMAXPROCS(0)
	if qw.OutDim < 512 || workers <= 1 {
		gemvNVFP4Rows(out, x, qw, 0, qw.OutDim)
		return
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
		go func(s, e int) {
			defer wg.Done()
			gemvNVFP4Rows(out, x, qw, s, e)
		}(start, end)
	}
	wg.Wait()
}

func gemvNVFP4Rows(out, x []float32, qw *NVFP4Weight, start, end int) {
	for row := start; row < end; row++ {
		sum := float32(0)
		for col := 0; col < qw.InDim; col++ {
			sum += nvfp4At(qw, row, col) * x[col]
		}
		out[row] = sum
	}
}

func countExceedsPackedNibbles(count, packedBytes int) bool {
	if packedBytes < 0 {
		return true
	}
	fullBytes := count / 2
	if fullBytes > packedBytes {
		return true
	}
	return count%2 != 0 && fullBytes >= packedBytes
}

func nvfp4At(qw *NVFP4Weight, row, col int) float32 {
	rowPacked := row * (qw.InDim / 2)
	rowScale := row * qw.Groups
	group := col / qw.GroupSize
	scale := DecodeF8E4M3(qw.WeightScale[rowScale+group]) * qw.WeightScale2
	b := qw.Weight[rowPacked+col/2]
	code := b & 0x0f
	if col%2 == 1 {
		code = b >> 4
	}
	return DecodeFP4E2M1(code) * scale
}
