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
	// The CPU NVFP4 path is primarily a correctness/reference fallback. Avoid
	// per-call goroutine allocation overhead for typical dense vectors; only
	// shard very large row counts where parallelism can amortize that cost.
	if qw.OutDim < 4096 || workers <= 1 {
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
	packedPerRow := qw.InDim / 2
	for row := start; row < end; row++ {
		rowPacked := row * packedPerRow
		rowScale := row * qw.Groups
		sum := float32(0)
		for group := 0; group < qw.Groups; group++ {
			scale := DecodeF8E4M3(qw.WeightScale[rowScale+group]) * qw.WeightScale2
			groupStart := group * qw.GroupSize
			groupEnd := groupStart + qw.GroupSize
			for col := groupStart; col < groupEnd; col += 2 {
				b := qw.Weight[rowPacked+col/2]
				sum += DecodeFP4E2M1(b&0x0f) * scale * x[col]
				if col+1 < groupEnd {
					sum += DecodeFP4E2M1(b>>4) * scale * x[col+1]
				}
			}
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
