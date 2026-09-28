package qwen

import (
	"fmt"
	"math"
)

// The pinned Transformers F32 recurrent kernel places epsilon inside rsqrt.
// Keep separate from the existing generation helper (sqrt(sum)+epsilon).
func qwen35BranchL2Norm(x []float32, eps float32) {
	var sum float32
	for _, v := range x {
		sum += v * v
	}
	scale := float32(1 / math.Sqrt(float64(sum+eps)))
	for i := range x {
		x[i] *= scale
	}
}

func qwen35BranchL2Heads(x []float32, heads, dim int, eps float32) error {
	if heads <= 0 || dim <= 0 || len(x)%heads != 0 || len(x)/heads != dim {
		return fmt.Errorf("qwen: invalid branch L2 geometry")
	}
	for h := 0; h < heads; h++ {
		qwen35BranchL2Norm(x[h*dim:(h+1)*dim], eps)
	}
	return nil
}
