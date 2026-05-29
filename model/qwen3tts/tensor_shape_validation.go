package qwen3tts

import (
	"fmt"
	"strings"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

type TensorShapeValidation struct {
	Valid  bool     `json:"valid"`
	Issues []string `json:"issues,omitempty"`
}

func ValidateTensorShapes(cfg ParsedConfig, infos map[string]safetensors.TensorInfo) TensorShapeValidation {
	v := TensorShapeValidation{Valid: true}
	for name, info := range infos {
		shape := info.Shape
		lower := strings.ToLower(name)
		switch {
		case strings.Contains(lower, "talker") && strings.Contains(lower, "q_proj"):
			if !matrixMatches(shape, cfg.TalkerHiddenSize, cfg.TalkerHiddenSize) {
				v.add(fmt.Sprintf("%s shape=%v want matrix using talker hidden=%d", name, shape, cfg.TalkerHiddenSize))
			}
		case strings.Contains(lower, "text_projection"):
			if len(shape) != 2 || !containsDim(shape, cfg.TalkerTextHiddenSize) || !containsDim(shape, cfg.TalkerHiddenSize) {
				v.add(fmt.Sprintf("%s shape=%v want dims text_hidden=%d and talker_hidden=%d", name, shape, cfg.TalkerTextHiddenSize, cfg.TalkerHiddenSize))
			}
		case strings.Contains(lower, "codec_embedding"):
			if len(shape) != 2 || shape[1] != cfg.CPHiddenSize {
				v.add(fmt.Sprintf("%s shape=%v want [*,%d]", name, shape, cfg.CPHiddenSize))
			}
		case strings.Contains(lower, "code_predictor") && strings.Contains(lower, "q_proj"):
			if !matrixMatches(shape, cfg.CPHiddenSize, cfg.CPHiddenSize) {
				v.add(fmt.Sprintf("%s shape=%v want matrix using code predictor hidden=%d", name, shape, cfg.CPHiddenSize))
			}
		case strings.Contains(lower, "codec_head"):
			if len(shape) != 2 || !containsDim(shape, cfg.TalkerHiddenSize) || !containsDim(shape, cfg.TalkerVocabSize) {
				v.add(fmt.Sprintf("%s shape=%v want dims talker_hidden=%d and vocab=%d", name, shape, cfg.TalkerHiddenSize, cfg.TalkerVocabSize))
			}
		}
	}
	return v
}

func matrixMatches(shape []int, rows, cols int) bool {
	return len(shape) == 2 && ((shape[0] == rows && shape[1] == cols) || (shape[0] == cols && shape[1] == rows))
}

func containsDim(shape []int, dim int) bool {
	for _, d := range shape {
		if d == dim {
			return true
		}
	}
	return false
}

func (v *TensorShapeValidation) add(issue string) {
	v.Valid = false
	v.Issues = append(v.Issues, issue)
}
