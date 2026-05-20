package format

import "fmt"

// ValidateQuantWeight checks an in-memory MLX quantized weight before use.
func ValidateQuantWeight(qw *QuantWeight) error {
	if qw == nil {
		return fmt.Errorf("nil MLX quant weight")
	}
	if qw.Bits <= 0 || qw.Bits > 32 || 32%qw.Bits != 0 {
		return fmt.Errorf("invalid MLX bits=%d", qw.Bits)
	}
	if qw.OutDim <= 0 || qw.InDim <= 0 || qw.GroupSize <= 0 || qw.Groups <= 0 {
		return fmt.Errorf("invalid MLX dims out=%d in=%d groupSize=%d groups=%d", qw.OutDim, qw.InDim, qw.GroupSize, qw.Groups)
	}
	packFactor := 32 / qw.Bits
	if qw.InDim%packFactor != 0 {
		return fmt.Errorf("MLX inDim=%d is not divisible by packFactor=%d", qw.InDim, packFactor)
	}
	if qw.InDim%qw.GroupSize != 0 || qw.Groups != qw.InDim/qw.GroupSize {
		return fmt.Errorf("MLX group layout mismatch inDim=%d groupSize=%d groups=%d", qw.InDim, qw.GroupSize, qw.Groups)
	}
	wantWeight, ok := checkedMulInt(qw.OutDim, qw.InDim/packFactor)
	if !ok {
		return fmt.Errorf("MLX weight size overflows out=%d in=%d packFactor=%d", qw.OutDim, qw.InDim, packFactor)
	}
	wantScale, ok := checkedMulInt(qw.OutDim, qw.Groups)
	if !ok {
		return fmt.Errorf("MLX scale/bias size overflows out=%d groups=%d", qw.OutDim, qw.Groups)
	}
	if len(qw.Weight) < wantWeight {
		return fmt.Errorf("MLX weight length=%d, expected at least %d", len(qw.Weight), wantWeight)
	}
	if len(qw.Scales) < wantScale || len(qw.Biases) < wantScale {
		return fmt.Errorf("MLX scale/bias length=%d/%d, expected at least %d", len(qw.Scales), len(qw.Biases), wantScale)
	}
	return nil
}
