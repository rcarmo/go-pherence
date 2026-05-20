package ops

import (
	"runtime"
	"sync"
)

// DequantNVFP4 dequantizes the observed ModelOpt NVFP4 layout to F32 using
// per-block E4M3 scales and the scalar weight_scale_2 multiplier. It returns
// [outDim, inDim] row-major F32 values and is intended as the correctness-first
// CPU reference path for tests and future fallback code.
func DequantNVFP4(qw *NVFP4Weight) []float32 {
	if err := ValidateNVFP4Weight(qw); err != nil {
		return nil
	}
	outLen, ok := checkedMulInt(qw.OutDim, qw.InDim)
	if !ok {
		return nil
	}
	out := make([]float32, outLen)
	if !DequantNVFP4To(out, qw) {
		return nil
	}
	return out
}

// DequantNVFP4To dequantizes into caller-owned storage. The output layout is
// [outDim, inDim] row-major. It returns false on malformed inputs or undersized
// output.
func DequantNVFP4To(out []float32, qw *NVFP4Weight) bool {
	if err := ValidateNVFP4Weight(qw); err != nil {
		return false
	}
	outLen, ok := checkedMulInt(qw.OutDim, qw.InDim)
	if !ok || len(out) < outLen {
		return false
	}
	// Dispatch hook kept explicit so AVX2/NEON dequant kernels can be wired
	// without changing callers. Scalar remains active until hasDequantAsm flips.
	dequantNVFP4Scalar(out[:outLen], qw)
	return true
}

func dequantNVFP4Scalar(out []float32, qw *NVFP4Weight) {
	workers := runtime.GOMAXPROCS(0)
	if qw.OutDim < 1024 || workers <= 1 {
		dequantNVFP4Rows(out, qw, 0, qw.OutDim)
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
			dequantNVFP4Rows(out, qw, s, e)
		}(start, end)
	}
	wg.Wait()
}

func dequantNVFP4Rows(out []float32, qw *NVFP4Weight, start, end int) {
	packedPerRow := qw.InDim / 2
	for row := start; row < end; row++ {
		rowPacked := row * packedPerRow
		rowScale := row * qw.Groups
		outRow := out[row*qw.InDim : (row+1)*qw.InDim]
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
				outRow[col] = DecodeFP4E2M1(code) * scale
			}
		}
	}
}
