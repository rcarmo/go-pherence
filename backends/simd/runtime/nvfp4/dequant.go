package nvfp4

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
	for row := 0; row < qw.OutDim; row++ {
		for col := 0; col < qw.InDim; col++ {
			out[row*qw.InDim+col] = nvfp4At(qw, row, col)
		}
	}
	return out
}
