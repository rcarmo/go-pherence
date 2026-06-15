package qwen3tts

import (
	"fmt"
	"strings"

	"github.com/rcarmo/go-pherence/loader/safetensors"
	"github.com/rcarmo/go-pherence/model/inspect"
	tensorinspect "github.com/rcarmo/go-pherence/model/internal/tensorinspect"
)

type TensorShapeValidation = tensorinspect.ShapeValidation

func ValidateTensorShapes(cfg ParsedConfig, infos map[string]safetensors.TensorInfo) TensorShapeValidation {
	v := TensorShapeValidation{Valid: true}
	for name, info := range infos {
		shape := info.Shape
		lower := strings.ToLower(name)
		switch {
		case strings.Contains(lower, "talker") && (strings.Contains(lower, "q_proj") || strings.Contains(lower, "o_proj")):
			if !inspect.MatrixMatches(shape, cfg.TalkerHiddenSize, cfg.TalkerHiddenSize) {
				v.Add(fmt.Sprintf("%s shape=%v want matrix using talker hidden=%d", name, shape, cfg.TalkerHiddenSize))
			}
		case strings.Contains(lower, "talker") && (strings.Contains(lower, "k_proj") || strings.Contains(lower, "v_proj")):
			kvWidth := cfg.TalkerNumKeyValueHeads * cfg.TalkerHeadDim
			if kvWidth > 0 && !inspect.MatrixMatches(shape, cfg.TalkerHiddenSize, kvWidth) {
				v.Add(fmt.Sprintf("%s shape=%v want matrix using talker hidden=%d and kv_width=%d", name, shape, cfg.TalkerHiddenSize, kvWidth))
			}
		case strings.Contains(lower, "talker") && (strings.Contains(lower, "gate_proj") || strings.Contains(lower, "up_proj") || strings.Contains(lower, "down_proj")):
			if !inspect.MatrixMatches(shape, cfg.TalkerHiddenSize, cfg.TalkerIntermediateSize) {
				v.Add(fmt.Sprintf("%s shape=%v want matrix using talker hidden=%d and intermediate=%d", name, shape, cfg.TalkerHiddenSize, cfg.TalkerIntermediateSize))
			}
		case strings.Contains(lower, "text_projection"):
			if len(shape) != 2 || !containsDim(shape, cfg.TalkerTextHiddenSize) || !containsDim(shape, cfg.TalkerHiddenSize) {
				v.Add(fmt.Sprintf("%s shape=%v want dims text_hidden=%d and talker_hidden=%d", name, shape, cfg.TalkerTextHiddenSize, cfg.TalkerHiddenSize))
			}
		case strings.Contains(lower, "codec_embedding"):
			if len(shape) != 2 || shape[1] != cfg.CPHiddenSize {
				v.Add(fmt.Sprintf("%s shape=%v want [*,%d]", name, shape, cfg.CPHiddenSize))
			}
		case strings.Contains(lower, "code_predictor") && (strings.Contains(lower, "q_proj") || strings.Contains(lower, "o_proj")):
			if !inspect.MatrixMatches(shape, cfg.CPHiddenSize, cfg.CPHiddenSize) {
				v.Add(fmt.Sprintf("%s shape=%v want matrix using code predictor hidden=%d", name, shape, cfg.CPHiddenSize))
			}
		case strings.Contains(lower, "code_predictor") && (strings.Contains(lower, "k_proj") || strings.Contains(lower, "v_proj")):
			kvWidth := cfg.CPNumKeyValueHeads * cfg.CPHeadDim
			if kvWidth > 0 && !inspect.MatrixMatches(shape, cfg.CPHiddenSize, kvWidth) {
				v.Add(fmt.Sprintf("%s shape=%v want matrix using code predictor hidden=%d and kv_width=%d", name, shape, cfg.CPHiddenSize, kvWidth))
			}
		case strings.Contains(lower, "code_predictor") && (strings.Contains(lower, "gate_proj") || strings.Contains(lower, "up_proj") || strings.Contains(lower, "down_proj")):
			if !inspect.MatrixMatches(shape, cfg.CPHiddenSize, cfg.CPIntermediateSize) {
				v.Add(fmt.Sprintf("%s shape=%v want matrix using code predictor hidden=%d and intermediate=%d", name, shape, cfg.CPHiddenSize, cfg.CPIntermediateSize))
			}
		case strings.Contains(lower, "codec_head"):
			if len(shape) != 2 || !containsDim(shape, cfg.TalkerHiddenSize) || !containsDim(shape, cfg.TalkerVocabSize) {
				v.Add(fmt.Sprintf("%s shape=%v want dims talker_hidden=%d and vocab=%d", name, shape, cfg.TalkerHiddenSize, cfg.TalkerVocabSize))
			}
		}
	}
	return v
}

func containsDim(shape []int, dim int) bool {
	for _, d := range shape {
		if d == dim {
			return true
		}
	}
	return false
}
