package lfm2

import (
	"fmt"
	"github.com/rcarmo/go-pherence/internal/checked"
	"strings"

	"github.com/rcarmo/go-pherence/loader/safetensors"
	"github.com/rcarmo/go-pherence/model/inspect"
	tensorinspect "github.com/rcarmo/go-pherence/model/internal/tensorinspect"
)

type TensorShapeValidation = tensorinspect.ShapeValidation

func ValidateTensorShapes(cfg Config, infos map[string]safetensors.TensorInfo) TensorShapeValidation {
	v := TensorShapeValidation{Valid: true}
	for name, info := range infos {
		shape := info.Shape
		if tensorElements(shape) <= 0 {
			v.Add(fmt.Sprintf("%s invalid/overflowing shape %v", name, shape))
			continue
		}
		lower := strings.ToLower(name)
		switch {
		case strings.Contains(lower, "embed_tokens"):
			vocab := cfg.VocabSize
			if vocab == 0 {
				vocab = 128000
			}
			if len(shape) != 2 || shape[0] != vocab || shape[1] != cfg.HiddenSize {
				v.Add(fmt.Sprintf("%s shape=%v want [%d,%d]", name, shape, vocab, cfg.HiddenSize))
			}
		case strings.Contains(lower, "embedding_norm") || strings.Contains(lower, "operator_norm") || strings.Contains(lower, "ffn_norm"):
			if len(shape) != 1 || shape[0] != cfg.HiddenSize {
				v.Add(fmt.Sprintf("%s shape=%v want [%d]", name, shape, cfg.HiddenSize))
			}
		case strings.Contains(lower, ".conv.in_proj.weight"):
			projected := sizeProduct(3, cfg.HiddenSize)
			if projected <= 0 || len(shape) != 2 || shape[0] != projected || shape[1] != cfg.HiddenSize {
				v.Add(fmt.Sprintf("%s shape=%v want [%d,%d]", name, shape, projected, cfg.HiddenSize))
			}
		case strings.Contains(lower, ".conv.in_proj.bias"):
			projected := sizeProduct(3, cfg.HiddenSize)
			if projected <= 0 || len(shape) != 1 || shape[0] != projected {
				v.Add(fmt.Sprintf("%s shape=%v want [%d]", name, shape, projected))
			}
		case strings.Contains(lower, ".conv.out_proj.weight"):
			if len(shape) != 2 || shape[0] != cfg.HiddenSize || shape[1] != cfg.HiddenSize {
				v.Add(fmt.Sprintf("%s shape=%v want [%d,%d]", name, shape, cfg.HiddenSize, cfg.HiddenSize))
			}
		case strings.Contains(lower, ".conv.out_proj.bias"):
			if len(shape) != 1 || shape[0] != cfg.HiddenSize {
				v.Add(fmt.Sprintf("%s shape=%v want [%d]", name, shape, cfg.HiddenSize))
			}
		case strings.Contains(lower, "q_proj") || strings.Contains(lower, "out_proj"):
			if cfg.HiddenSize > 0 && !inspect.MatrixMatches(shape, cfg.HiddenSize, cfg.HiddenSize) {
				v.Add(fmt.Sprintf("%s shape=%v want matrix using hidden=%d", name, shape, cfg.HiddenSize))
			}
		case strings.Contains(lower, "k_proj") || strings.Contains(lower, "v_proj"):
			kvWidth, ok := checked.MulInt(cfg.NumKeyValueHeads, cfg.HeadDim)
			if !ok {
				v.Add("KV width overflows")
				continue
			}
			if cfg.HiddenSize > 0 && kvWidth > 0 && !inspect.MatrixMatches(shape, cfg.HiddenSize, kvWidth) {
				v.Add(fmt.Sprintf("%s shape=%v want matrix using hidden=%d and kv_width=%d", name, shape, cfg.HiddenSize, kvWidth))
			}
		case strings.Contains(lower, ".conv.conv.weight"):
			want, ok := checked.MulInt(cfg.HiddenSize, cfg.ConvLCache)
			if !ok {
				v.Add("conv size overflows")
				continue
			}
			if cfg.HiddenSize > 0 && cfg.ConvLCache > 0 && (tensorElements(shape) != want || len(shape) != 3 || shape[0] != cfg.HiddenSize || shape[1] != 1 || shape[2] != cfg.ConvLCache) {
				v.Add(fmt.Sprintf("%s shape=%v want [%d,1,%d]", name, shape, cfg.HiddenSize, cfg.ConvLCache))
			}
		case strings.Contains(lower, ".conv.conv.bias"):
			if len(shape) != 1 || shape[0] != cfg.HiddenSize {
				v.Add(fmt.Sprintf("%s shape=%v want [%d]", name, shape, cfg.HiddenSize))
			}
		case strings.Contains(lower, "q_layernorm") || strings.Contains(lower, "k_layernorm"):
			if len(shape) != 1 || shape[0] != cfg.HeadDim {
				v.Add(fmt.Sprintf("%s shape=%v want [%d]", name, shape, cfg.HeadDim))
			}
		case strings.Contains(lower, ".experts.gate_up_proj"):
			width := sizeProduct(2, cfg.MoEIntermediateSize)
			if width <= 0 || len(shape) != 3 || shape[0] != cfg.NumExperts || shape[1] != width || shape[2] != cfg.HiddenSize {
				v.Add(fmt.Sprintf("%s shape=%v want [%d,%d,%d]", name, shape, cfg.NumExperts, width, cfg.HiddenSize))
			}
		case strings.Contains(lower, ".experts.down_proj"):
			if len(shape) != 3 || shape[0] != cfg.NumExperts || shape[1] != cfg.HiddenSize || shape[2] != cfg.MoEIntermediateSize {
				v.Add(fmt.Sprintf("%s shape=%v want [%d,%d,%d]", name, shape, cfg.NumExperts, cfg.HiddenSize, cfg.MoEIntermediateSize))
			}
		case strings.Contains(lower, ".feed_forward.gate.weight") || strings.Contains(lower, ".router.weight"):
			if len(shape) != 2 || shape[0] != cfg.NumExperts || shape[1] != cfg.HiddenSize {
				v.Add(fmt.Sprintf("%s shape=%v want [%d,%d]", name, shape, cfg.NumExperts, cfg.HiddenSize))
			}
		case strings.Contains(lower, ".expert_bias"):
			if len(shape) != 1 || shape[0] != cfg.NumExperts {
				v.Add(fmt.Sprintf("%s shape=%v want [%d]", name, shape, cfg.NumExperts))
			}
		case strings.Contains(lower, "lm_head"):
			vocab := cfg.VocabSize
			if vocab == 0 {
				vocab = 128000
			}
			if len(shape) != 2 || shape[0] != vocab || shape[1] != cfg.HiddenSize {
				v.Add(fmt.Sprintf("%s shape=%v want [%d,%d]", name, shape, vocab, cfg.HiddenSize))
			}
		}
	}
	return v
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
		var ok bool
		n, ok = checked.MulInt(n, d)
		if !ok {
			return 0
		}
	}
	return n
}
