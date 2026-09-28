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
	if err := cfg.Validate(); err != nil {
		v.Add(err.Error())
		return v
	}
	for name, info := range infos {
		shape := info.Shape
		lower := strings.ToLower(name)
		switch {
		case strings.Contains(lower, "talker") && !strings.Contains(lower, "code_predictor") && strings.Contains(lower, "q_proj"):
			queryWidth := sizeProduct(cfg.TalkerNumAttentionHeads, cfg.TalkerHeadDim)
			if queryWidth <= 0 || !inspect.MatrixMatches(shape, cfg.TalkerHiddenSize, queryWidth) {
				v.Add(fmt.Sprintf("%s shape=%v want matrix using talker hidden=%d and query_width=%d", name, shape, cfg.TalkerHiddenSize, queryWidth))
			}
		case strings.Contains(lower, "talker") && !strings.Contains(lower, "code_predictor") && strings.Contains(lower, "o_proj"):
			queryWidth := sizeProduct(cfg.TalkerNumAttentionHeads, cfg.TalkerHeadDim)
			if queryWidth <= 0 || !inspect.MatrixMatches(shape, queryWidth, cfg.TalkerHiddenSize) {
				v.Add(fmt.Sprintf("%s shape=%v want matrix using query_width=%d and talker hidden=%d", name, shape, queryWidth, cfg.TalkerHiddenSize))
			}
		case strings.Contains(lower, "talker") && !strings.Contains(lower, "code_predictor") && (strings.Contains(lower, "k_proj") || strings.Contains(lower, "v_proj")):
			kvWidth := sizeProduct(cfg.TalkerNumKeyValueHeads, cfg.TalkerHeadDim)
			if kvWidth <= 0 || !inspect.MatrixMatches(shape, cfg.TalkerHiddenSize, kvWidth) {
				v.Add(fmt.Sprintf("%s shape=%v want matrix using talker hidden=%d and kv_width=%d", name, shape, cfg.TalkerHiddenSize, kvWidth))
			}
		case strings.Contains(lower, "talker") && !strings.Contains(lower, "code_predictor") && (strings.Contains(lower, "gate_proj") || strings.Contains(lower, "up_proj") || strings.Contains(lower, "down_proj")):
			if !inspect.MatrixMatches(shape, cfg.TalkerHiddenSize, cfg.TalkerIntermediateSize) {
				v.Add(fmt.Sprintf("%s shape=%v want matrix using talker hidden=%d and intermediate=%d", name, shape, cfg.TalkerHiddenSize, cfg.TalkerIntermediateSize))
			}
		case strings.Contains(lower, "text_projection") && strings.HasSuffix(lower, ".weight"):
			if strings.Contains(lower, "linear_fc1") {
				if !inspect.MatrixMatches(shape, cfg.TalkerTextHiddenSize, cfg.TalkerTextHiddenSize) {
					v.Add(fmt.Sprintf("%s shape=%v want text projection fc1 [%d,%d]", name, shape, cfg.TalkerTextHiddenSize, cfg.TalkerTextHiddenSize))
				}
			} else if !inspect.MatrixMatches(shape, cfg.TalkerTextHiddenSize, cfg.TalkerHiddenSize) {
				v.Add(fmt.Sprintf("%s shape=%v want dims text_hidden=%d and talker_hidden=%d", name, shape, cfg.TalkerTextHiddenSize, cfg.TalkerHiddenSize))
			}
		case strings.Contains(lower, "text_projection") && strings.HasSuffix(lower, ".bias"):
			want := cfg.TalkerHiddenSize
			if strings.Contains(lower, "linear_fc1") {
				want = cfg.TalkerTextHiddenSize
			}
			if len(shape) != 1 || shape[0] != want {
				v.Add(fmt.Sprintf("%s shape=%v want bias [%d]", name, shape, want))
			}
		case strings.Contains(lower, "codec_embedding"):
			width := cfg.CPHiddenSize
			if strings.Contains(lower, "talker.model.codec_embedding") {
				width = cfg.TalkerHiddenSize
			}
			if len(shape) != 2 || shape[1] != width {
				v.Add(fmt.Sprintf("%s shape=%v want [*,%d]", name, shape, width))
			}
		case strings.Contains(lower, "code_predictor") && strings.Contains(lower, "q_proj"):
			queryWidth := sizeProduct(cfg.CPNumAttentionHeads, cfg.CPHeadDim)
			if queryWidth <= 0 || !inspect.MatrixMatches(shape, cfg.CPHiddenSize, queryWidth) {
				v.Add(fmt.Sprintf("%s shape=%v want matrix using code predictor hidden=%d and query_width=%d", name, shape, cfg.CPHiddenSize, queryWidth))
			}
		case strings.Contains(lower, "code_predictor") && strings.Contains(lower, "o_proj"):
			queryWidth := sizeProduct(cfg.CPNumAttentionHeads, cfg.CPHeadDim)
			if queryWidth <= 0 || !inspect.MatrixMatches(shape, queryWidth, cfg.CPHiddenSize) {
				v.Add(fmt.Sprintf("%s shape=%v want matrix using query_width=%d and code predictor hidden=%d", name, shape, queryWidth, cfg.CPHiddenSize))
			}
		case strings.Contains(lower, "code_predictor") && (strings.Contains(lower, "k_proj") || strings.Contains(lower, "v_proj")):
			kvWidth := sizeProduct(cfg.CPNumKeyValueHeads, cfg.CPHeadDim)
			if kvWidth <= 0 || !inspect.MatrixMatches(shape, cfg.CPHiddenSize, kvWidth) {
				v.Add(fmt.Sprintf("%s shape=%v want matrix using code predictor hidden=%d and kv_width=%d", name, shape, cfg.CPHiddenSize, kvWidth))
			}
		case strings.Contains(lower, "code_predictor") && (strings.Contains(lower, "gate_proj") || strings.Contains(lower, "up_proj") || strings.Contains(lower, "down_proj")):
			if !inspect.MatrixMatches(shape, cfg.CPHiddenSize, cfg.CPIntermediateSize) {
				v.Add(fmt.Sprintf("%s shape=%v want matrix using code predictor hidden=%d and intermediate=%d", name, shape, cfg.CPHiddenSize, cfg.CPIntermediateSize))
			}
		case strings.Contains(lower, "codec_head"):
			if !inspect.MatrixMatches(shape, cfg.TalkerHiddenSize, cfg.TalkerVocabSize) {
				v.Add(fmt.Sprintf("%s shape=%v want dims talker_hidden=%d and vocab=%d", name, shape, cfg.TalkerHiddenSize, cfg.TalkerVocabSize))
			}
		}
	}
	return v
}
