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

func (v *TensorShapeValidation) add(issue string) {
	v.Valid = false
	v.Issues = append(v.Issues, issue)
}
