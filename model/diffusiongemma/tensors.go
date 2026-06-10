package diffusiongemma

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

type TensorGroup string

const (
	TensorDecoderEmbedding TensorGroup = "decoder_embedding"
	TensorDecoderLayer     TensorGroup = "decoder_layer"
	TensorMoE              TensorGroup = "moe"
	TensorAttention        TensorGroup = "attention"
	TensorRouter           TensorGroup = "router"
	TensorVision           TensorGroup = "vision"
	TensorProjector        TensorGroup = "projector"
	TensorNorm             TensorGroup = "norm"
	TensorOther            TensorGroup = "other"
)

type TensorInventory struct {
	IndexPath string                   `json:"index_path"`
	Total     int                      `json:"total"`
	Shards    int                      `json:"shards"`
	Groups    map[TensorGroup]int      `json:"groups"`
	Examples  map[TensorGroup][]string `json:"examples,omitempty"`
}

type hfSafetensorsIndex struct {
	WeightMap map[string]string `json:"weight_map"`
}

func TensorInventoryFromModelDir(modelDir string) (TensorInventory, bool, error) {
	path := filepath.Join(modelDir, "model.safetensors.index.json")
	inv, err := TensorInventoryFromIndex(path)
	if err != nil {
		if os.IsNotExist(err) {
			return TensorInventory{}, false, nil
		}
		return TensorInventory{}, false, err
	}
	return inv, true, nil
}

func TensorInventoryFromIndex(path string) (TensorInventory, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return TensorInventory{}, err
	}
	var idx hfSafetensorsIndex
	if err := json.Unmarshal(b, &idx); err != nil {
		return TensorInventory{}, err
	}
	inv := TensorInventory{IndexPath: path, Groups: map[TensorGroup]int{}, Examples: map[TensorGroup][]string{}}
	shards := map[string]bool{}
	names := make([]string, 0, len(idx.WeightMap))
	for name, shard := range idx.WeightMap {
		names = append(names, name)
		shards[shard] = true
	}
	sort.Strings(names)
	inv.Total = len(names)
	inv.Shards = len(shards)
	for _, name := range names {
		g := ClassifyTensorName(name)
		inv.Groups[g]++
		if len(inv.Examples[g]) < 8 {
			inv.Examples[g] = append(inv.Examples[g], name)
		}
	}
	return inv, nil
}

func ClassifyTensorName(name string) TensorGroup {
	switch {
	case strings.Contains(name, ".embed_tokens.") || strings.HasSuffix(name, ".embed_tokens.weight"):
		return TensorDecoderEmbedding
	case strings.Contains(name, "vision") || strings.HasPrefix(name, "model.encoder."):
		return TensorVision
	case strings.Contains(name, "projector") || strings.Contains(name, "multi_modal"):
		return TensorProjector
	case strings.Contains(name, ".experts."):
		return TensorMoE
	case strings.Contains(name, ".router"):
		return TensorRouter
	case strings.Contains(name, ".self_attn.") || strings.Contains(name, ".cross_attn."):
		return TensorAttention
	case strings.Contains(name, "norm"):
		return TensorNorm
	case strings.Contains(name, ".layers."):
		return TensorDecoderLayer
	default:
		return TensorOther
	}
}

type TensorReadiness struct {
	RuntimeReady         bool     `json:"runtime_ready"`
	TextReady            bool     `json:"text_ready"`
	VisionInventoryReady bool     `json:"vision_inventory_ready"`
	MissingRequired      []string `json:"missing_required,omitempty"`
	MissingLayerTensors  int      `json:"missing_layer_tensors,omitempty"`
	ExpectedLayerTensors int      `json:"expected_layer_tensors,omitempty"`
	ObservedLayerTensors int      `json:"observed_layer_tensors,omitempty"`
	ExpectedTextLayers   int      `json:"expected_text_layers"`
	ObservedTextLayers   int      `json:"observed_text_layers"`
}

var diffusionGemmaRequiredGlobals = []string{
	"model.decoder.embed_tokens.weight",
	"model.decoder.norm.weight",
	"model.decoder.self_conditioning.down_proj.weight",
	"model.decoder.self_conditioning.gate_proj.weight",
	"model.decoder.self_conditioning.pre_norm.weight",
	"model.decoder.self_conditioning.up_proj.weight",
}

var diffusionGemmaRequiredLayerSuffixes = []string{
	"input_layernorm.weight",
	"layer_scalar",
	"mlp.down_proj.weight",
	"mlp.gate_proj.weight",
	"mlp.up_proj.weight",
	"post_attention_layernorm.weight",
	"post_feedforward_layernorm.weight",
	"pre_feedforward_layernorm.weight",
	"router.per_expert_scale",
	"router.proj.weight",
	"router.scale",
	"self_attn.k_norm.weight",
	"self_attn.k_proj.weight",
	"self_attn.o_proj.weight",
	"self_attn.q_norm.weight",
	"self_attn.q_proj.weight",
	"self_attn.v_proj.weight",
	"experts.down_proj",
	"experts.gate_up_proj",
}

func TensorReadinessFromInventory(inv TensorInventory, shape Shape) TensorReadiness {
	present := map[string]bool{}
	// Examples are intentionally capped; use IndexPath when available for full readiness.
	if inv.IndexPath != "" {
		if names, err := tensorNamesFromIndex(inv.IndexPath); err == nil {
			for _, name := range names {
				present[name] = true
			}
		}
	}
	out := TensorReadiness{ExpectedTextLayers: shape.TextLayers}
	for _, name := range diffusionGemmaRequiredGlobals {
		if !present[name] {
			out.MissingRequired = append(out.MissingRequired, name)
		}
	}
	observedLayers := map[int]bool{}
	for layer := 0; layer < shape.TextLayers; layer++ {
		base := "model.decoder.layers." + itoa(layer) + "."
		for _, suffix := range requiredLayerSuffixesForType(layerTypeAt(shape.LayerTypes, layer)) {
			out.ExpectedLayerTensors++
			if present[base+suffix] {
				out.ObservedLayerTensors++
				observedLayers[layer] = true
			} else {
				out.MissingLayerTensors++
				if len(out.MissingRequired) < 32 {
					out.MissingRequired = append(out.MissingRequired, base+suffix)
				}
			}
		}
	}
	out.ObservedTextLayers = len(observedLayers)
	out.TextReady = len(out.MissingRequired) == 0 && out.MissingLayerTensors == 0 && out.ObservedTextLayers == shape.TextLayers
	out.VisionInventoryReady = inv.Groups[TensorVision] > 0
	out.RuntimeReady = false
	return out
}

func requiredLayerSuffixesForType(layerType string) []string {
	if layerType != "full_attention" {
		return diffusionGemmaRequiredLayerSuffixes
	}
	out := make([]string, 0, len(diffusionGemmaRequiredLayerSuffixes)-1)
	for _, suffix := range diffusionGemmaRequiredLayerSuffixes {
		if suffix == "self_attn.v_proj.weight" {
			continue
		}
		out = append(out, suffix)
	}
	return out
}

func layerTypeAt(types []string, i int) string {
	if i < 0 || i >= len(types) {
		return ""
	}
	return types[i]
}

func tensorNamesFromIndex(path string) ([]string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var idx hfSafetensorsIndex
	if err := json.Unmarshal(b, &idx); err != nil {
		return nil, err
	}
	names := make([]string, 0, len(idx.WeightMap))
	for name := range idx.WeightMap {
		names = append(names, name)
	}
	sort.Strings(names)
	return names, nil
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}
