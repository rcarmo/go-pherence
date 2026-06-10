package q4

import (
	"fmt"
	"github.com/rcarmo/go-pherence/internal/checked"
)

func ValidateGemv(out, x []float32, qweight, qzeros, gIdx []int32, scales []float32, inDim, outDim int, sym bool) error {
	if err := Validate(qweight, qzeros, gIdx, scales, inDim, outDim, sym); err != nil {
		return fmt.Errorf("GemvQ4 GPTQ validation: %w", err)
	}
	if len(out) < outDim {
		return fmt.Errorf("GemvQ4 out length=%d, expected at least %d", len(out), outDim)
	}
	if len(x) < inDim {
		return fmt.Errorf("GemvQ4 x length=%d, expected at least %d", len(x), inDim)
	}
	return nil
}

func ValidateGemvSym(out, x []float32, qweight, gIdx []int32, scales []float32, inDim, outDim int) error {
	if err := ValidateGemv(out, x, qweight, nil, gIdx, scales, inDim, outDim, true); err != nil {
		return fmt.Errorf("GemvQ4Sym GPTQ validation: %w", err)
	}
	return nil
}

func ValidateSym(qweight, gIdx []int32, scales []float32, inFeatures, outFeatures int) error {
	return Validate(qweight, nil, gIdx, scales, inFeatures, outFeatures, true)
}

func Validate(qweight, qzeros, gIdx []int32, scales []float32, inFeatures, outFeatures int, sym bool) error {
	if inFeatures <= 0 || outFeatures <= 0 {
		return fmt.Errorf("invalid GPTQ dims in=%d out=%d", inFeatures, outFeatures)
	}
	if inFeatures%8 != 0 {
		return fmt.Errorf("GPTQ inFeatures=%d is not divisible by 8", inFeatures)
	}
	if !sym && outFeatures%8 != 0 {
		return fmt.Errorf("GPTQ outFeatures=%d is not divisible by 8 for qzeros", outFeatures)
	}
	if _, ok := checked.MulInt(inFeatures, outFeatures); !ok {
		return fmt.Errorf("GPTQ output size overflows for in=%d out=%d", inFeatures, outFeatures)
	}
	wantQWeight, ok := checked.MulInt(inFeatures/8, outFeatures)
	if !ok {
		return fmt.Errorf("GPTQ qweight size overflows for in=%d out=%d", inFeatures, outFeatures)
	}
	if len(qweight) < wantQWeight {
		return fmt.Errorf("GPTQ qweight length=%d, expected at least %d", len(qweight), wantQWeight)
	}
	if len(gIdx) < inFeatures {
		return fmt.Errorf("GPTQ g_idx length=%d, expected at least %d", len(gIdx), inFeatures)
	}

	maxGroup := -1
	for i := 0; i < inFeatures; i++ {
		g := int(gIdx[i])
		if g < 0 {
			return fmt.Errorf("GPTQ g_idx[%d]=%d is negative", i, g)
		}
		if g > maxGroup {
			maxGroup = g
		}
	}
	if maxGroup < 0 {
		return fmt.Errorf("GPTQ g_idx has no groups")
	}

	wantScales, ok := checked.MulInt(maxGroup+1, outFeatures)
	if !ok {
		return fmt.Errorf("GPTQ scales size overflows for %d groups and out=%d", maxGroup+1, outFeatures)
	}
	if len(scales) < wantScales {
		return fmt.Errorf("GPTQ scales length=%d, expected at least %d for %d groups", len(scales), wantScales, maxGroup+1)
	}
	if !sym {
		wantQZeros, ok := checked.MulInt(maxGroup+1, outFeatures/8)
		if !ok {
			return fmt.Errorf("GPTQ qzeros size overflows for %d groups and out=%d", maxGroup+1, outFeatures)
		}
		if len(qzeros) < wantQZeros {
			return fmt.Errorf("GPTQ qzeros length=%d, expected at least %d for %d groups", len(qzeros), wantQZeros, maxGroup+1)
		}
	}
	return nil
}
