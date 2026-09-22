package minicpmv

import (
	"fmt"
	"strings"

	"github.com/rcarmo/go-pherence/loader/config"
	"github.com/rcarmo/go-pherence/loader/safetensors"
	"github.com/rcarmo/go-pherence/model/inspect"
	tensorinspect "github.com/rcarmo/go-pherence/model/internal/tensorinspect"
)

type TensorShapeValidation = tensorinspect.ShapeValidation

func ValidateTensorShapes(summary config.MiniCPMVSummary, infos map[string]safetensors.TensorInfo) TensorShapeValidation {
	v := TensorShapeValidation{Valid: true}
	for name, info := range infos {
		shape := info.Shape
		switch ClassifyTensorName(name) {
		case TensorTextEmbedding:
			if summary.HiddenSize > 0 && (len(shape) != 2 || shape[1] != summary.HiddenSize) {
				v.Add(fmt.Sprintf("%s shape=%v want [*,%d]", name, shape, summary.HiddenSize))
			}
		case TensorTextLMHead:
			if summary.HiddenSize > 0 && (len(shape) != 2 || shape[1] != summary.HiddenSize) {
				v.Add(fmt.Sprintf("%s shape=%v want [*,%d]", name, shape, summary.HiddenSize))
			}
		case TensorResampler:
			validateResamplerShape(&v, name, shape, summary)
		case TensorProjector:
			if summary.HiddenSize > 0 && summary.VisionHiddenSize > 0 && len(shape) == 2 && !containsDim(shape, summary.HiddenSize) && !containsDim(shape, summary.VisionHiddenSize) {
				v.Add(fmt.Sprintf("%s shape=%v want dims involving hidden=%d or vision_hidden=%d", name, shape, summary.HiddenSize, summary.VisionHiddenSize))
			}
		case TensorVisionTower:
			validateVisionTensorShape(&v, name, shape, summary)
		case TensorAudioEncoder:
			validateAudioTensorShape(&v, name, shape, summary)
		case TensorTextLayer:
			validateTextLayerShape(&v, name, shape, summary)
		}
	}
	return v
}

func validateVisionTensorShape(v *TensorShapeValidation, name string, shape []int, summary config.MiniCPMVSummary) {
	lower := strings.ToLower(name)
	if strings.Contains(lower, "patch") && strings.Contains(lower, "weight") && summary.PatchSize > 0 && len(shape) >= 3 && !containsDim(shape, summary.PatchSize) {
		v.Add(fmt.Sprintf("%s shape=%v want patch_size=%d in patch embedding", name, shape, summary.PatchSize))
		return
	}
	if strings.Contains(lower, "pos_embed") && summary.VisionHiddenSize > 0 && (len(shape) < 2 || shape[len(shape)-1] != summary.VisionHiddenSize) {
		v.Add(fmt.Sprintf("%s shape=%v want final vision_hidden=%d", name, shape, summary.VisionHiddenSize))
		return
	}
	if strings.Contains(lower, ".attn.qkv.weight") && summary.VisionHiddenSize > 0 {
		if len(shape) != 2 || shape[0] != 3*summary.VisionHiddenSize || shape[1] != summary.VisionHiddenSize {
			v.Add(fmt.Sprintf("%s shape=%v want [%d,%d]", name, shape, 3*summary.VisionHiddenSize, summary.VisionHiddenSize))
		}
	}
}

func validateTextLayerShape(v *TensorShapeValidation, name string, shape []int, summary config.MiniCPMVSummary) {
	lower := strings.ToLower(name)
	if summary.HiddenSize <= 0 {
		return
	}
	if strings.HasSuffix(lower, ".bias") {
		want := 0
		switch {
		case strings.Contains(lower, "q_proj"):
			want = summary.Heads * summary.HeadDim
		case strings.Contains(lower, "k_proj") || strings.Contains(lower, "v_proj"):
			want = summary.KVHeads * summary.HeadDim
		}
		if want > 0 && (len(shape) != 1 || shape[0] != want) {
			v.Add(fmt.Sprintf("%s shape=%v want bias [%d]", name, shape, want))
		}
		return
	}
	if strings.Contains(lower, "q_proj") || strings.Contains(lower, "o_proj") {
		queryWidth := summary.Heads * summary.HeadDim
		if queryWidth > 0 && !inspect.MatrixMatches(shape, summary.HiddenSize, queryWidth) {
			v.Add(fmt.Sprintf("%s shape=%v want matrix using hidden=%d and query_width=%d", name, shape, summary.HiddenSize, queryWidth))
		}
		return
	}
	if strings.Contains(lower, "k_proj") || strings.Contains(lower, "v_proj") {
		kvWidth := summary.KVHeads * summary.HeadDim
		if kvWidth > 0 && !inspect.MatrixMatches(shape, summary.HiddenSize, kvWidth) {
			v.Add(fmt.Sprintf("%s shape=%v want matrix using hidden=%d and kv_width=%d", name, shape, summary.HiddenSize, kvWidth))
		}
		return
	}
	if strings.Contains(lower, "gate_proj") || strings.Contains(lower, "up_proj") || strings.Contains(lower, "down_proj") {
		if summary.IntermediateSize > 0 && !inspect.MatrixMatches(shape, summary.HiddenSize, summary.IntermediateSize) {
			v.Add(fmt.Sprintf("%s shape=%v want matrix using hidden=%d and intermediate=%d", name, shape, summary.HiddenSize, summary.IntermediateSize))
		}
	}
}

func validateAudioTensorShape(v *TensorShapeValidation, name string, shape []int, summary config.MiniCPMVSummary) {
	lower := strings.ToLower(name)
	if summary.AudioHiddenSize <= 0 {
		return
	}
	if strings.HasPrefix(lower, "audio_projection_layer.linear1.weight") && summary.HiddenSize > 0 {
		if len(shape) != 2 || shape[0] != summary.HiddenSize || shape[1] != summary.AudioHiddenSize {
			v.Add(fmt.Sprintf("%s shape=%v want [%d,%d]", name, shape, summary.HiddenSize, summary.AudioHiddenSize))
		}
		return
	}
	if strings.HasPrefix(lower, "audio_projection_layer.linear2.weight") && summary.HiddenSize > 0 {
		if len(shape) != 2 || shape[0] != summary.HiddenSize || shape[1] != summary.HiddenSize {
			v.Add(fmt.Sprintf("%s shape=%v want [%d,%d]", name, shape, summary.HiddenSize, summary.HiddenSize))
		}
		return
	}
	if strings.HasPrefix(lower, "audio_projection_layer") && strings.HasSuffix(lower, ".bias") && summary.HiddenSize > 0 {
		if len(shape) != 1 || shape[0] != summary.HiddenSize {
			v.Add(fmt.Sprintf("%s shape=%v want [%d]", name, shape, summary.HiddenSize))
		}
		return
	}
	if strings.Contains(lower, "conv") && strings.Contains(lower, "weight") {
		if len(shape) < 2 || !containsDim(shape, summary.AudioHiddenSize) {
			v.Add(fmt.Sprintf("%s shape=%v want audio_hidden=%d in convolution weight", name, shape, summary.AudioHiddenSize))
		}
		return
	}
	if strings.Contains(lower, "q_proj") || strings.Contains(lower, "k_proj") || strings.Contains(lower, "v_proj") || strings.Contains(lower, "out_proj") || strings.Contains(lower, "o_proj") {
		if !inspect.MatrixMatches(shape, summary.AudioHiddenSize, summary.AudioHiddenSize) {
			v.Add(fmt.Sprintf("%s shape=%v want matrix using audio_hidden=%d", name, shape, summary.AudioHiddenSize))
		}
		return
	}
	if strings.Contains(lower, "fc") || strings.Contains(lower, "gate_proj") || strings.Contains(lower, "up_proj") || strings.Contains(lower, "down_proj") {
		if summary.AudioIntermediateSize > 0 && !inspect.MatrixMatches(shape, summary.AudioHiddenSize, summary.AudioIntermediateSize) {
			v.Add(fmt.Sprintf("%s shape=%v want matrix using audio_hidden=%d and audio_intermediate=%d", name, shape, summary.AudioHiddenSize, summary.AudioIntermediateSize))
		}
	}
}

func validateResamplerShape(v *TensorShapeValidation, name string, shape []int, summary config.MiniCPMVSummary) {
	lower := strings.ToLower(name)
	if strings.Contains(lower, "query") && summary.NumQuery > 0 && summary.HiddenSize > 0 {
		if len(shape) < 2 || shape[len(shape)-2] != summary.NumQuery || shape[len(shape)-1] != summary.HiddenSize {
			v.Add(fmt.Sprintf("%s shape=%v want [...,%d,%d]", name, shape, summary.NumQuery, summary.HiddenSize))
		}
		return
	}
	if strings.Contains(lower, "pos_embed") && summary.NumQuery > 0 {
		if !containsDim(shape, summary.NumQuery) {
			v.Add(fmt.Sprintf("%s shape=%v want num_query=%d", name, shape, summary.NumQuery))
		}
		return
	}
	if strings.Contains(lower, "kv_proj") && summary.HiddenSize > 0 && summary.VisionHiddenSize > 0 {
		if !inspect.MatrixMatches(shape, summary.VisionHiddenSize, summary.HiddenSize) {
			v.Add(fmt.Sprintf("%s shape=%v want matrix vision_hidden=%d hidden=%d", name, shape, summary.VisionHiddenSize, summary.HiddenSize))
		}
	}
}

func containsDim(shape []int, dim int) bool {
	for _, d := range shape {
		if d == dim {
			return true
		}
	}
	return false
}
