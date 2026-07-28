package nvfp4

import "github.com/rcarmo/go-pherence/internal/checked"

// GemmNVFP4 computes out[batch,outDim] = x[batch,inDim] @ W^T directly from
// packed NVFP4 weights while preserving GemvNVFP4To as the batch=1 contract.
func GemmNVFP4(out, x []float32, batch int, qw *NVFP4Weight) bool {
	if batch <= 0 || ValidateNVFP4Weight(qw) != nil {
		return false
	}
	xLen, okX := checked.MulInt(batch, qw.InDim)
	outLen, okOut := checked.MulInt(batch, qw.OutDim)
	if !okX || !okOut || len(x) < xLen || len(out) < outLen {
		return false
	}
	if batch == 1 {
		return GemvNVFP4To(out[:qw.OutDim], x[:qw.InDim], qw)
	}
	if RuntimeCapabilities().HasGemv {
		return gemmNVFP4Accelerated(out[:outLen], x[:xLen], batch, qw)
	}
	return gemmNVFP4Portable(out[:outLen], x[:xLen], batch, qw)
}
