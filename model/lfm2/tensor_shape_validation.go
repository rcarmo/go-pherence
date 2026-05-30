package lfm2

import (
	"fmt"
	"strings"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

type TensorShapeValidation struct {
	Valid  bool     `json:"valid"`
	Issues []string `json:"issues,omitempty"`
}

func ValidateTensorShapes(cfg Config, infos map[string]safetensors.TensorInfo) TensorShapeValidation {
	v := TensorShapeValidation{Valid: true}
	for name, info := range infos {
		shape := info.Shape
		lower := strings.ToLower(name)
		switch {
		case strings.Contains(lower, "embed_tokens"):
			if len(shape) != 2 || shape[1] != cfg.HiddenSize {
				v.add(fmt.Sprintf("%s shape=%v want [*,%d]", name, shape, cfg.HiddenSize))
			}
		case strings.Contains(lower, "q_proj") || strings.Contains(lower, "o_proj"):
			if cfg.HiddenSize > 0 && !matrixMatches(shape, cfg.HiddenSize, cfg.HiddenSize) {
				v.add(fmt.Sprintf("%s shape=%v want matrix using hidden=%d", name, shape, cfg.HiddenSize))
			}
		case strings.Contains(lower, "k_proj") || strings.Contains(lower, "v_proj"):
			kvWidth := cfg.NumKeyValueHeads * cfg.HeadDim
			if cfg.HiddenSize > 0 && kvWidth > 0 && !matrixMatches(shape, cfg.HiddenSize, kvWidth) {
				v.add(fmt.Sprintf("%s shape=%v want matrix using hidden=%d and kv_width=%d", name, shape, cfg.HiddenSize, kvWidth))
			}
		case strings.Contains(lower, "conv") && strings.Contains(lower, "weight"):
			want := cfg.HiddenSize * cfg.ConvLCache
			if cfg.HiddenSize > 0 && cfg.ConvLCache > 0 && tensorElements(shape) != want {
				v.add(fmt.Sprintf("%s shape=%v want %d conv kernel elements", name, shape, want))
			}
		case strings.Contains(lower, "router") || strings.Contains(lower, ".gate") || strings.HasSuffix(lower, "gate.weight"):
			if len(shape) != 2 || shape[0] != cfg.NumExperts || shape[1] != cfg.HiddenSize {
				v.add(fmt.Sprintf("%s shape=%v want [%d,%d]", name, shape, cfg.NumExperts, cfg.HiddenSize))
			}
		case strings.Contains(lower, "experts"):
			if len(shape) == 2 && shape[1] != cfg.HiddenSize && shape[0] != cfg.HiddenSize && shape[0] != cfg.MoEIntermediateSize && shape[1] != cfg.MoEIntermediateSize {
				v.add(fmt.Sprintf("%s shape=%v does not reference hidden=%d or moe_intermediate=%d", name, shape, cfg.HiddenSize, cfg.MoEIntermediateSize))
			}
		case strings.Contains(lower, "lm_head"):
			if len(shape) != 2 || shape[1] != cfg.HiddenSize {
				v.add(fmt.Sprintf("%s shape=%v want [*,%d]", name, shape, cfg.HiddenSize))
			}
		}
	}
	return v
}

func matrixMatches(shape []int, rows, cols int) bool {
	return len(shape) == 2 && ((shape[0] == rows && shape[1] == cols) || (shape[0] == cols && shape[1] == rows))
}

func tensorElements(shape []int) int {
	if len(shape) == 0 {
		return 0
	}
	n := 1
	for _, d := range shape {
		if d <= 0 {
			return 0
		}
		n *= d
	}
	return n
}

func (v *TensorShapeValidation) add(issue string) {
	v.Valid = false
	v.Issues = append(v.Issues, issue)
}
