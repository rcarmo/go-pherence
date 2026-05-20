package ops

import (
	"runtime"
	"sync"
)

// GemvNVFP4 performs a correctness-first matrix-vector multiply directly from
// packed NVFP4 weights: out[outDim] = W_nvfp4[outDim, inDim] · x[inDim].
func GemvNVFP4(out, x []float32, qw *NVFP4Weight) { _ = GemvNVFP4To(out, x, qw) }

// GemvNVFP4To performs packed NVFP4 GEMV into caller-owned output and reports
// malformed inputs while preserving GemvNVFP4's no-allocation behavior.
func GemvNVFP4To(out, x []float32, qw *NVFP4Weight) bool {
	if err := ValidateNVFP4Weight(qw); err != nil || len(out) < qw.OutDim || len(x) < qw.InDim {
		return false
	}
	// Dispatch hook kept explicit so AVX2/NEON GEMV kernels can be wired without
	// changing callers. Scalar/reference path remains active until hasGemvAsm flips.
	workers := runtime.GOMAXPROCS(0)
	// The CPU NVFP4 path is primarily a correctness/reference fallback. Avoid
	// per-call goroutine allocation overhead for typical dense vectors; only
	// shard very large row counts where parallelism can amortize that cost.
	if qw.OutDim < 4096 || workers <= 1 {
		gemvNVFP4Rows(out, x, qw, 0, qw.OutDim)
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
		go func(s, e int) {
			defer wg.Done()
			gemvNVFP4Rows(out, x, qw, s, e)
		}(start, end)
	}
	wg.Wait()
	return true
}

// GemvNVFP4Reference is the exact single-thread scalar reference path for
// parity tests. Future AVX2/NEON/native implementations should compare against
// this function rather than against another optimized path.
func GemvNVFP4Reference(out, x []float32, qw *NVFP4Weight) bool {
	if err := ValidateNVFP4Weight(qw); err != nil || len(out) < qw.OutDim || len(x) < qw.InDim {
		return false
	}
	gemvNVFP4Rows(out, x, qw, 0, qw.OutDim)
	return true
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
			for col := groupStart; col < groupEnd; col++ {
				b := qw.Weight[rowPacked+col/2]
				code := b & 0x0f
				if col%2 == 1 {
					code = b >> 4
				}
				sum += DecodeFP4E2M1(code) * scale * x[col]
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
