package ideogram4

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
)

type TensorInventory struct {
	Total               int      `json:"total"`
	Global              int      `json:"global"`
	Layer               int      `json:"layer"`
	LayerCount          int      `json:"layer_count"`
	MissingGlobals      []string `json:"missing_globals,omitempty"`
	MissingLayerTensors []string `json:"missing_layer_tensors,omitempty"`
	FP8Weights          int      `json:"fp8_weights"`
	FP8Scales           int      `json:"fp8_scales"`
	HasFP8Scales        bool     `json:"has_fp8_scales"`
}

var RequiredGlobalTensors = []string{
	"input_proj.weight",
	"input_proj.bias",
	"llm_cond_norm.weight",
	"llm_cond_proj.weight",
	"llm_cond_proj.bias",
	"t_embedding.mlp_in.weight",
	"t_embedding.mlp_in.bias",
	"t_embedding.mlp_out.weight",
	"t_embedding.mlp_out.bias",
	"adaln_proj.weight",
	"adaln_proj.bias",
	"embed_image_indicator.weight",
	"final_layer.adaln_modulation.weight",
	"final_layer.adaln_modulation.bias",
	"final_layer.linear.weight",
	"final_layer.linear.bias",
}

var RequiredLayerTensorSuffixes = []string{
	"adaln_modulation.weight",
	"adaln_modulation.bias",
	"attention.norm_q.weight",
	"attention.norm_k.weight",
	"attention.qkv.weight",
	"attention.o.weight",
	"attention_norm1.weight",
	"attention_norm2.weight",
	"feed_forward.w1.weight",
	"feed_forward.w2.weight",
	"feed_forward.w3.weight",
	"ffn_norm1.weight",
	"ffn_norm2.weight",
}

func SummarizeTensorNames(names []string, expectedLayers int) TensorInventory {
	set := make(map[string]bool, len(names))
	layers := map[int]bool{}
	inv := TensorInventory{Total: len(names)}
	for _, name := range names {
		set[name] = true
		if strings.HasPrefix(name, "layers.") {
			inv.Layer++
			parts := strings.Split(name, ".")
			if len(parts) > 1 {
				if idx, err := strconv.Atoi(parts[1]); err == nil {
					layers[idx] = true
				}
			}
		} else {
			inv.Global++
		}
		if strings.HasSuffix(name, ".weight") {
			if set[name+"_scale"] || containsName(names, name+"_scale") {
				inv.FP8Weights++
			}
		}
		if strings.HasSuffix(name, ".weight_scale") {
			inv.FP8Scales++
		}
	}
	inv.LayerCount = len(layers)
	inv.HasFP8Scales = inv.FP8Scales > 0
	for _, required := range RequiredGlobalTensors {
		if !set[required] {
			inv.MissingGlobals = append(inv.MissingGlobals, required)
		}
	}
	if expectedLayers > 0 {
		for i := 0; i < expectedLayers; i++ {
			if !layers[i] {
				inv.MissingLayerTensors = append(inv.MissingLayerTensors, fmt.Sprintf("layers.%d.*", i))
				continue
			}
			for _, suffix := range RequiredLayerTensorSuffixes {
				name := fmt.Sprintf("layers.%d.%s", i, suffix)
				if !set[name] {
					inv.MissingLayerTensors = append(inv.MissingLayerTensors, name)
				}
			}
		}
	}
	sort.Strings(inv.MissingGlobals)
	sort.Strings(inv.MissingLayerTensors)
	return inv
}

func (inv TensorInventory) RuntimeReady(expectedLayers int) bool {
	if inv.Total == 0 || len(inv.MissingGlobals) > 0 || len(inv.MissingLayerTensors) > 0 {
		return false
	}
	if expectedLayers > 0 && inv.LayerCount != expectedLayers {
		return false
	}
	return inv.HasFP8Scales
}

func containsName(names []string, target string) bool {
	for _, name := range names {
		if name == target {
			return true
		}
	}
	return false
}
