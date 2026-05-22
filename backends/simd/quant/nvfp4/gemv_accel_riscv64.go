//go:build riscv64

package nvfp4

import simd "github.com/rcarmo/go-pherence/backends/simd/runtime"

func gemvNVFP4Accelerated(out, x []float32, qw *NVFP4Weight) {
	wrow := make([]float32, qw.InDim)
	for row := 0; row < qw.OutDim; row++ {
		dequantNVFP4Row(wrow, qw, row)
		out[row] = simd.Sdot(x[:qw.InDim], wrow)
	}
}

func dequantNVFP4Row(out []float32, qw *NVFP4Weight, row int) {
	rowPacked := row * (qw.InDim / 2)
	rowScale := row * qw.Groups
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
			out[col] = DecodeFP4E2M1(code) * scale
		}
	}
}
