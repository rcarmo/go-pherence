package mlx

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
	for b := 0; b < batch; b++ {
		xRow := x[b*qw.InDim : (b+1)*qw.InDim]
		outRow := out[b*qw.OutDim : (b+1)*qw.OutDim]
		Gemv(outRow, xRow, qw)
	}
	return true
}
