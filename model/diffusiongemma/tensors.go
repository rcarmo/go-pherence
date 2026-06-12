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
	IndexPath       string                   `json:"index_path"`
	Total           int                      `json:"total"`
	Shards          int                      `json:"shards"`
	TotalParameters int64                    `json:"total_parameters,omitempty"`
	TotalSizeBytes  int64                    `json:"total_size_bytes,omitempty"`
	Groups          map[TensorGroup]int      `json:"groups"`
	Examples        map[TensorGroup][]string `json:"examples,omitempty"`
}

type hfSafetensorsIndex struct {
	Metadata struct {
		TotalParameters int64 `json:"total_parameters"`
		TotalSize       int64 `json:"total_size"`
	} `json:"metadata"`
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
	inv := TensorInventory{IndexPath: path, TotalParameters: idx.Metadata.TotalParameters, TotalSizeBytes: idx.Metadata.TotalSize, Groups: map[TensorGroup]int{}, Examples: map[TensorGroup][]string{}}
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
	"post_feedforward_layernorm_1.weight",
	"post_feedforward_layernorm_2.weight",
	"pre_feedforward_layernorm.weight",
	"pre_feedforward_layernorm_2.weight",
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
			} else if isExpertSuffix(suffix) && present[base+"experts.0.gate_proj.weight"] {
				// Per-expert FP8 format: experts.{N}.{gate,up,down}_proj instead of fused
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
	idx, err := readSafetensorsIndex(path)
	if err != nil {
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

type TensorHandle struct {
	Name      string      `json:"name"`
	Shard     string      `json:"shard,omitempty"`
	Group     TensorGroup `json:"group"`
	Required  bool        `json:"required"`
	PerExpert bool        `json:"per_expert,omitempty"`
}

type LayerTensorPlan struct {
	Layer   int            `json:"layer"`
	Type    string         `json:"type,omitempty"`
	Handles []TensorHandle `json:"handles"`
}

type TextTensorPlan struct {
	IndexPath string            `json:"index_path"`
	Globals   []TensorHandle    `json:"globals"`
	Layers    []LayerTensorPlan `json:"layers"`
	Ready     bool              `json:"ready"`
	Missing   []string          `json:"missing,omitempty"`
}

func TextTensorPlanFromModelDir(modelDir string, shape Shape) (TextTensorPlan, bool, error) {
	path := filepath.Join(modelDir, "model.safetensors.index.json")
	plan, err := TextTensorPlanFromIndex(path, shape)
	if err != nil {
		if os.IsNotExist(err) {
			return TextTensorPlan{}, false, nil
		}
		return TextTensorPlan{}, false, err
	}
	return plan, true, nil
}

func TextTensorPlanFromIndex(path string, shape Shape) (TextTensorPlan, error) {
	weightMap, err := tensorWeightMapFromIndex(path)
	if err != nil {
		return TextTensorPlan{}, err
	}
	plan := TextTensorPlan{IndexPath: path, Ready: true}
	for _, name := range diffusionGemmaRequiredGlobals {
		h := TensorHandle{Name: name, Shard: weightMap[name], Group: ClassifyTensorName(name), Required: true}
		if h.Shard == "" {
			plan.Ready = false
			plan.Missing = append(plan.Missing, name)
		}
		plan.Globals = append(plan.Globals, h)
	}
	for layer := 0; layer < shape.TextLayers; layer++ {
		lt := layerTypeAt(shape.LayerTypes, layer)
		lp := LayerTensorPlan{Layer: layer, Type: lt}
		base := "model.decoder.layers." + itoa(layer) + "."
		for _, suffix := range requiredLayerSuffixesForType(lt) {
			name := base + suffix
			h := TensorHandle{Name: name, Shard: weightMap[name], Group: ClassifyTensorName(name), Required: true}
			if h.Shard == "" && isExpertSuffix(suffix) {
				// Per-expert FP8 format fallback
				altName := base + "experts.0.gate_proj.weight"
				if weightMap[altName] != "" {
					h.Shard = weightMap[altName]
					h.Name = altName
					h.PerExpert = true
				}
			}
			if h.Shard == "" {
				plan.Ready = false
				if len(plan.Missing) < 64 {
					plan.Missing = append(plan.Missing, name)
				}
			}
			lp.Handles = append(lp.Handles, h)
		}
		plan.Layers = append(plan.Layers, lp)
	}
	return plan, nil
}

func tensorWeightMapFromIndex(path string) (map[string]string, error) {
	idx, err := readSafetensorsIndex(path)
	if err != nil {
		return nil, err
	}
	return idx.WeightMap, nil
}

func readSafetensorsIndex(path string) (hfSafetensorsIndex, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return hfSafetensorsIndex{}, err
	}
	var idx hfSafetensorsIndex
	if err := json.Unmarshal(b, &idx); err != nil {
		return hfSafetensorsIndex{}, err
	}
	return idx, nil
}

type ShardAvailability struct {
	IndexPath          string   `json:"index_path"`
	ExpectedShards     int      `json:"expected_shards"`
	PresentShards      int      `json:"present_shards"`
	MissingShards      []string `json:"missing_shards,omitempty"`
	Ready              bool     `json:"ready"`
	PresentPercent     float64  `json:"present_percent"`
	ExpectedBytes      int64    `json:"expected_bytes"`
	PresentBytes       int64    `json:"present_bytes"`
	PresentBytePercent float64  `json:"present_byte_percent"`
}

func ShardAvailabilityFromModelDir(modelDir string) (ShardAvailability, bool, error) {
	path := filepath.Join(modelDir, "model.safetensors.index.json")
	index, err := readSafetensorsIndex(path)
	if err != nil {
		if os.IsNotExist(err) {
			return ShardAvailability{}, false, nil
		}
		return ShardAvailability{}, false, err
	}
	shards := map[string]bool{}
	for _, shard := range index.WeightMap {
		shards[shard] = true
	}
	names := make([]string, 0, len(shards))
	for shard := range shards {
		names = append(names, shard)
	}
	sort.Strings(names)
	out := ShardAvailability{IndexPath: path, ExpectedShards: len(names), ExpectedBytes: index.Metadata.TotalSize, Ready: true}
	for _, shard := range names {
		if info, err := os.Stat(filepath.Join(modelDir, shard)); err == nil {
			out.PresentShards++
			out.PresentBytes += info.Size()
			continue
		} else if err != nil && !os.IsNotExist(err) {
			return ShardAvailability{}, false, err
		}
		out.Ready = false
		out.MissingShards = append(out.MissingShards, shard)
	}
	if out.ExpectedShards > 0 {
		out.PresentPercent = 100 * float64(out.PresentShards) / float64(out.ExpectedShards)
	}
	if out.ExpectedBytes > 0 {
		out.PresentBytePercent = 100 * float64(out.PresentBytes) / float64(out.ExpectedBytes)
	}
	return out, true, nil
}

func isExpertSuffix(suffix string) bool {
	return suffix == "experts.down_proj" || suffix == "experts.gate_up_proj"
}
