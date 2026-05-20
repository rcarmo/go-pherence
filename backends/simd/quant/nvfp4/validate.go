package nvfp4

import "fmt"

// ValidateNVFP4Weight checks the observed NVFP4 packed-weight and scale layout.
func ValidateNVFP4Weight(qw *NVFP4Weight) error {
	if qw == nil {
		return fmt.Errorf("nil NVFP4 quant weight")
	}
	if qw.OutDim <= 0 || qw.InDim <= 0 || qw.GroupSize <= 0 || qw.Groups <= 0 {
		return fmt.Errorf("invalid NVFP4 dims out=%d in=%d groupSize=%d groups=%d", qw.OutDim, qw.InDim, qw.GroupSize, qw.Groups)
	}
	if qw.InDim%2 != 0 {
		return fmt.Errorf("NVFP4 inDim=%d is not divisible by packed FP4 factor=2", qw.InDim)
	}
	if qw.InDim%qw.GroupSize != 0 || qw.Groups != qw.InDim/qw.GroupSize {
		return fmt.Errorf("NVFP4 group layout mismatch inDim=%d groupSize=%d groups=%d", qw.InDim, qw.GroupSize, qw.Groups)
	}
	wantWeight, ok := checkedMulInt(qw.OutDim, qw.InDim/2)
	if !ok {
		return fmt.Errorf("NVFP4 weight size overflows out=%d in=%d", qw.OutDim, qw.InDim)
	}
	wantScale, ok := checkedMulInt(qw.OutDim, qw.Groups)
	if !ok {
		return fmt.Errorf("NVFP4 scale size overflows out=%d groups=%d", qw.OutDim, qw.Groups)
	}
	if len(qw.Weight) < wantWeight {
		return fmt.Errorf("NVFP4 weight length=%d, expected at least %d", len(qw.Weight), wantWeight)
	}
	if len(qw.WeightScale) < wantScale {
		return fmt.Errorf("NVFP4 weight_scale length=%d, expected at least %d", len(qw.WeightScale), wantScale)
	}
	return nil
}
