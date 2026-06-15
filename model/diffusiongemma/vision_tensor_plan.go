package diffusiongemma

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

var diffusionGemmaVisionGlobals = []string{
	"model.encoder.vision_tower.patch_embedder.input_proj.weight",
	"model.encoder.vision_tower.patch_embedder.position_embedding_table",
	"model.encoder.vision_tower.std_bias",
	"model.encoder.vision_tower.std_scale",
	"model.encoder.embed_vision.embedding_projection.weight",
}

var diffusionGemmaVisionLayerSuffixes = []string{
	"input_layernorm.weight",
	"self_attn.q_proj.linear.weight",
	"self_attn.k_proj.linear.weight",
	"self_attn.v_proj.linear.weight",
	"self_attn.o_proj.linear.weight",
	"self_attn.q_norm.weight",
	"self_attn.k_norm.weight",
	"post_attention_layernorm.weight",
	"pre_feedforward_layernorm.weight",
	"mlp.gate_proj.linear.weight",
	"mlp.up_proj.linear.weight",
	"mlp.down_proj.linear.weight",
	"post_feedforward_layernorm.weight",
}

type VisionTensorPlan struct {
	IndexPath string            `json:"index_path"`
	Globals   []TensorHandle    `json:"globals"`
	Layers    []LayerTensorPlan `json:"layers"`
	Ready     bool              `json:"ready"`
	Missing   []string          `json:"missing,omitempty"`
}

func VisionTensorPlanFromModelDir(modelDir string, shape Shape) (VisionTensorPlan, bool, error) {
	path := filepath.Join(modelDir, "model.safetensors.index.json")
	plan, err := VisionTensorPlanFromIndex(path, shape)
	if err != nil {
		if os.IsNotExist(err) {
			return VisionTensorPlan{}, false, nil
		}
		return VisionTensorPlan{}, false, err
	}
	return plan, true, nil
}

func VisionTensorPlanFromIndex(path string, shape Shape) (VisionTensorPlan, error) {
	weightMap, err := tensorWeightMapFromIndex(path)
	if err != nil {
		return VisionTensorPlan{}, err
	}
	plan := VisionTensorPlan{IndexPath: path, Ready: true}
	for _, name := range diffusionGemmaVisionGlobals {
		h := TensorHandle{Name: name, Shard: weightMap[name], Group: ClassifyTensorName(name), Required: true}
		if h.Shard == "" {
			plan.Ready = false
			plan.Missing = append(plan.Missing, name)
		}
		plan.Globals = append(plan.Globals, h)
	}
	for layer := 0; layer < shape.VisionLayers; layer++ {
		lp := LayerTensorPlan{Layer: layer, Type: "vision"}
		base := "model.encoder.vision_tower.encoder.layers." + itoa(layer) + "."
		for _, suffix := range diffusionGemmaVisionLayerSuffixes {
			name := base + suffix
			h := TensorHandle{Name: name, Shard: weightMap[name], Group: ClassifyTensorName(name), Required: true}
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

func ValidateVisionTensorPlanShapes(plan VisionTensorPlan, infos map[string]safetensors.TensorInfo, shape Shape) []string {
	var mismatches []string
	check := func(name string, dims ...int) {
		info, ok := infos[name]
		if !ok {
			mismatches = append(mismatches, name+": missing metadata")
			return
		}
		if len(info.Shape) != len(dims) {
			mismatches = append(mismatches, fmt.Sprintf("%s: rank=%d want %d shape=%v", name, len(info.Shape), len(dims), info.Shape))
			return
		}
		for i, want := range dims {
			if want >= 0 && info.Shape[i] != want {
				mismatches = append(mismatches, fmt.Sprintf("%s: shape=%v want dim[%d]=%d", name, info.Shape, i, want))
				return
			}
		}
	}
	hidden := shape.VisionHiddenSize
	textHidden := shape.TextHiddenSize
	patchIn := 3 * shape.PatchSize * shape.PatchSize
	check("model.encoder.vision_tower.patch_embedder.input_proj.weight", hidden, patchIn)
	check("model.encoder.vision_tower.patch_embedder.position_embedding_table", 2, -1, hidden)
	check("model.encoder.vision_tower.std_bias", hidden)
	check("model.encoder.vision_tower.std_scale", hidden)
	check("model.encoder.embed_vision.embedding_projection.weight", textHidden, hidden)
	headDim := -1
	if shape.VisionHeads > 0 {
		headDim = hidden / shape.VisionHeads
	}
	for layer := 0; layer < shape.VisionLayers; layer++ {
		base := "model.encoder.vision_tower.encoder.layers." + itoa(layer) + "."
		check(base+"input_layernorm.weight", hidden)
		check(base+"self_attn.q_proj.linear.weight", hidden, hidden)
		check(base+"self_attn.k_proj.linear.weight", hidden, hidden)
		check(base+"self_attn.v_proj.linear.weight", hidden, hidden)
		check(base+"self_attn.o_proj.linear.weight", hidden, hidden)
		check(base+"self_attn.q_norm.weight", headDim)
		check(base+"self_attn.k_norm.weight", headDim)
		check(base+"post_attention_layernorm.weight", hidden)
		check(base+"pre_feedforward_layernorm.weight", hidden)
		gateInfo, gateOK := infos[base+"mlp.gate_proj.linear.weight"]
		upInfo, upOK := infos[base+"mlp.up_proj.linear.weight"]
		downInfo, downOK := infos[base+"mlp.down_proj.linear.weight"]
		if !gateOK || !upOK || !downOK {
			if !gateOK {
				mismatches = append(mismatches, base+"mlp.gate_proj.linear.weight: missing metadata")
			}
			if !upOK {
				mismatches = append(mismatches, base+"mlp.up_proj.linear.weight: missing metadata")
			}
			if !downOK {
				mismatches = append(mismatches, base+"mlp.down_proj.linear.weight: missing metadata")
			}
			continue
		}
		gate := gateInfo.Shape
		up := upInfo.Shape
		down := downInfo.Shape
		if len(gate) != 2 || gate[1] != hidden {
			mismatches = append(mismatches, fmt.Sprintf("%smlp.gate_proj.linear.weight: shape=%v want [*,%d]", base, gate, hidden))
		}
		if len(up) != 2 || len(gate) != 2 || up[0] != gate[0] || up[1] != hidden {
			mismatches = append(mismatches, fmt.Sprintf("%smlp.up_proj.linear.weight: shape=%v want [%d,%d]", base, up, firstDim(gate), hidden))
		}
		if len(down) != 2 || len(gate) != 2 || down[0] != hidden || down[1] != gate[0] {
			mismatches = append(mismatches, fmt.Sprintf("%smlp.down_proj.linear.weight: shape=%v want [%d,%d]", base, down, hidden, firstDim(gate)))
		}
		check(base+"post_feedforward_layernorm.weight", hidden)
	}
	return mismatches
}

func firstDim(shape []int) int {
	if len(shape) == 0 {
		return -1
	}
	return shape[0]
}
