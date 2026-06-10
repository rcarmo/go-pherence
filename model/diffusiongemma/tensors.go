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
