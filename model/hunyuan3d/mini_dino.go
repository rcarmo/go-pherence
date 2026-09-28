package hunyuan3d

import (
	"fmt"
	"slices"
	"strings"
)

// MiniDINOConditioner binds the metadata of the standard Hunyuan3D-2mini
// DINOv2 giant conditioner. It does not load weights or execute the encoder.
// TensorRef values remain tied to the checkpoint owned by Model.
type MiniDINOConditioner struct {
	Tensors []TensorRef
}

const miniDINOPrefix = "conditioner.main_image_encoder.model."
const miniDINOLayers = 40
const miniDINOWidth = 1536

type miniDINOTensorSpec struct {
	name  string
	shape []int
}

var miniDINOGlobalSpecs = []miniDINOTensorSpec{
	{"embeddings.cls_token", []int{1, 1, miniDINOWidth}},
	{"embeddings.mask_token", []int{1, miniDINOWidth}},
	{"embeddings.patch_embeddings.projection.bias", []int{miniDINOWidth}},
	{"embeddings.patch_embeddings.projection.weight", []int{miniDINOWidth, 3, 14, 14}},
	{"embeddings.position_embeddings", []int{1, 1370, miniDINOWidth}},
	{"layernorm.bias", []int{miniDINOWidth}},
	{"layernorm.weight", []int{miniDINOWidth}},
}

// The mini checkpoint has 40 copies of these exact per-layer names and shapes.
// Its MLP input is twice the intermediate width for a gated activation.
var miniDINOLayerSpecs = []miniDINOTensorSpec{
	{"attention.attention.key.bias", []int{miniDINOWidth}},
	{"attention.attention.key.weight", []int{miniDINOWidth, miniDINOWidth}},
	{"attention.attention.query.bias", []int{miniDINOWidth}},
	{"attention.attention.query.weight", []int{miniDINOWidth, miniDINOWidth}},
	{"attention.attention.value.bias", []int{miniDINOWidth}},
	{"attention.attention.value.weight", []int{miniDINOWidth, miniDINOWidth}},
	{"attention.output.dense.bias", []int{miniDINOWidth}},
	{"attention.output.dense.weight", []int{miniDINOWidth, miniDINOWidth}},
	{"layer_scale1.lambda1", []int{miniDINOWidth}},
	{"layer_scale2.lambda1", []int{miniDINOWidth}},
	{"mlp.weights_in.bias", []int{8192}},
	{"mlp.weights_in.weight", []int{8192, miniDINOWidth}},
	{"mlp.weights_out.bias", []int{miniDINOWidth}},
	{"mlp.weights_out.weight", []int{miniDINOWidth, 4096}},
	{"norm1.bias", []int{miniDINOWidth}},
	{"norm1.weight", []int{miniDINOWidth}},
	{"norm2.bias", []int{miniDINOWidth}},
	{"norm2.weight", []int{miniDINOWidth}},
}

// BindMiniDINOConditioner checks every name, F16 dtype and shape of the
// standard single-view mini conditioner. It rejects other variants rather than
// silently binding a partially compatible encoder.
func (m *Model) BindMiniDINOConditioner() (*MiniDINOConditioner, error) {
	if m == nil || m.checkpoint == nil || m.Config.Conditioner.Params.MainImageEncoder.Type != "DinoImageEncoder" || m.Config.Conditioner.Params.AdditionalImageEncoder.Type != "" || m.Condition.ImageSize != 518 || m.Condition.PatchSize != 14 || m.Condition.Channels != 3 || m.Shape.ContextInDim != miniDINOWidth || !miniDINOConfigMatches(m.Config.Conditioner.Params.MainImageEncoder.Kwargs) {
		return nil, fmt.Errorf("hunyuan3d mini DINO: unsupported conditioner configuration")
	}
	refs := m.Tensors.Conditioner
	count := len(miniDINOGlobalSpecs) + miniDINOLayers*len(miniDINOLayerSpecs)
	if len(refs) != count {
		return nil, fmt.Errorf("hunyuan3d mini DINO: tensor count=%d want=%d", len(refs), count)
	}
	bound := &MiniDINOConditioner{Tensors: make([]TensorRef, 0, count)}
	bind := func(name string, shape []int) error {
		full := miniDINOPrefix + name
		ref, ok := refs[full]
		if !ok || ref.Name != full || ref.file != m.checkpoint || ref.DType != "F16" || !slices.Equal(ref.Shape, shape) {
			return fmt.Errorf("hunyuan3d mini DINO: missing or incompatible tensor %q", full)
		}
		ref.Shape = slices.Clone(ref.Shape)
		bound.Tensors = append(bound.Tensors, ref)
		return nil
	}
	for _, spec := range miniDINOGlobalSpecs {
		if err := bind(spec.name, spec.shape); err != nil {
			return nil, err
		}
	}
	for layer := 0; layer < miniDINOLayers; layer++ {
		for _, spec := range miniDINOLayerSpecs {
			if err := bind(fmt.Sprintf("encoder.layer.%d.%s", layer, spec.name), spec.shape); err != nil {
				return nil, err
			}
		}
	}
	// Count plus the complete expected set also rejects all extra variant keys.
	for name := range refs {
		if !strings.HasPrefix(name, miniDINOPrefix) {
			return nil, fmt.Errorf("hunyuan3d mini DINO: unexpected tensor %q", name)
		}
	}
	return bound, nil
}

func miniDINOConfigMatches(kwargs map[string]any) bool {
	config, ok := kwargs["config"].(map[string]any)
	if !ok || kwargs["image_size"] != 518 {
		return false
	}
	for name, want := range map[string]int{
		"hidden_size": miniDINOWidth, "num_hidden_layers": miniDINOLayers,
		"num_attention_heads": 24, "num_channels": 3, "patch_size": 14,
		"image_size": 518,
	} {
		if config[name] != want {
			return false
		}
	}
	return config["use_swiglu_ffn"] == true && config["qkv_bias"] == true
}
