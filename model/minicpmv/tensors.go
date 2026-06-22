package minicpmv

import (
	"sort"
	"strings"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

type TensorGroup string

const (
	TensorTextEmbedding TensorGroup = "text_embedding"
	TensorTextLayer     TensorGroup = "text_layer"
	TensorTextLMHead    TensorGroup = "text_lm_head"
	TensorVisionTower   TensorGroup = "vision_tower"
	TensorAudioEncoder  TensorGroup = "audio_encoder"
	TensorResampler     TensorGroup = "resampler"
	TensorProjector     TensorGroup = "projector"
	TensorNorm          TensorGroup = "norm"
	TensorOther         TensorGroup = "other"
)

type TensorInventory struct {
	Total    int                      `json:"total"`
	Groups   map[TensorGroup]int      `json:"groups"`
	Examples map[TensorGroup][]string `json:"examples,omitempty"`
}

type TensorReadiness struct {
	HasTextEmbedding bool `json:"has_text_embedding"`
	HasTextLayers    bool `json:"has_text_layers"`
	HasLMHead        bool `json:"has_lm_head"`
	HasVisionTower   bool `json:"has_vision_tower"`
	HasAudioEncoder  bool `json:"has_audio_encoder"`
	HasResampler     bool `json:"has_resampler"`
	HasProjector     bool `json:"has_projector"`
	MetadataReady    bool `json:"metadata_ready"`
	RuntimeReady     bool `json:"runtime_ready"`
}

func TensorInventoryFromModelDir(modelDir string) (TensorInventory, bool, error) {
	names, err := safetensors.NamesFrom(modelDir, "")
	if err != nil {
		return TensorInventory{}, false, nil
	}
	return SummarizeTensors(names), true, nil
}

func SummarizeTensors(names []string) TensorInventory {
	sorted := append([]string(nil), names...)
	sort.Strings(sorted)
	inv := TensorInventory{Total: len(sorted), Groups: map[TensorGroup]int{}, Examples: map[TensorGroup][]string{}}
	for _, name := range sorted {
		g := ClassifyTensorName(name)
		inv.Groups[g]++
		if len(inv.Examples[g]) < 8 {
			inv.Examples[g] = append(inv.Examples[g], name)
		}
	}
	return inv
}

func ClassifyTensorName(name string) TensorGroup {
	n := strings.ToLower(name)
	switch {
	case strings.Contains(n, "resampler") || strings.Contains(n, "query_tokens") || strings.Contains(n, "pos_embed"):
		return TensorResampler
	case strings.Contains(n, "audio") || strings.Contains(n, "whisper") || strings.Contains(n, "apm") || strings.Contains(n, "audio_tower") || strings.Contains(n, "audio_encoder"):
		return TensorAudioEncoder
	case strings.Contains(n, "vision_tower") || strings.Contains(n, "vpm") || strings.Contains(n, "vision_model") || strings.Contains(n, "clip") || strings.Contains(n, "siglip") || strings.Contains(n, "eva"):
		return TensorVisionTower
	case strings.Contains(n, "mm_projector") || strings.Contains(n, "multi_modal_projector") || strings.Contains(n, "vision_proj"):
		return TensorProjector
	case strings.Contains(n, "embed_tokens.weight") || strings.HasSuffix(n, "tok_embeddings.weight"):
		return TensorTextEmbedding
	case strings.Contains(n, "lm_head.weight") || strings.Contains(n, "output.weight"):
		return TensorTextLMHead
	case strings.Contains(n, "norm.weight") || strings.Contains(n, "ln_") || strings.Contains(n, "layernorm"):
		return TensorNorm
	case strings.Contains(n, ".layers.") || strings.Contains(n, ".blocks.") || strings.Contains(n, "model.layers"):
		return TensorTextLayer
	default:
		return TensorOther
	}
}

func TensorReadinessFromInventory(inv TensorInventory) TensorReadiness {
	out := TensorReadiness{
		HasTextEmbedding: inv.Groups[TensorTextEmbedding] > 0,
		HasTextLayers:    inv.Groups[TensorTextLayer] > 0,
		HasLMHead:        inv.Groups[TensorTextLMHead] > 0,
		HasVisionTower:   inv.Groups[TensorVisionTower] > 0,
		HasAudioEncoder:  inv.Groups[TensorAudioEncoder] > 0,
		HasResampler:     inv.Groups[TensorResampler] > 0,
		HasProjector:     inv.Groups[TensorProjector] > 0,
	}
	out.MetadataReady = out.HasTextEmbedding && out.HasTextLayers && out.HasVisionTower && out.HasResampler
	out.RuntimeReady = false
	return out
}
