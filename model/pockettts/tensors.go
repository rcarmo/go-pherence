package pockettts

import (
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

type TensorInventory struct {
	Total, FlowLM, Mimi, Other int
	FlowLayers, FlowBlocks     int
	MimiEncoderLayers          int
	MimiDecoderLayers          int
	DTypes                     map[string]int
	Missing                    []string
	ShapesValid                bool
	Issues                     []string
}

func InspectTensorInfos(cfg Config, infos map[string]safetensors.TensorInfo) TensorInventory {
	inv := TensorInventory{Total: len(infos), DTypes: map[string]int{}, ShapesValid: true}
	flowLayers, flowBlocks, encLayers, decLayers := map[int]bool{}, map[int]bool{}, map[int]bool{}, map[int]bool{}
	for name, info := range infos {
		inv.DTypes[info.DType]++
		switch {
		case strings.HasPrefix(name, "flow_lm."):
			inv.FlowLM++
		case strings.HasPrefix(name, "mimi."):
			inv.Mimi++
		default:
			inv.Other++
		}
		markLayer(name, "flow_lm.transformer.layers.", flowLayers)
		markLayer(name, "flow_lm.flow_net.res_blocks.", flowBlocks)
		markLayer(name, "mimi.encoder_transformer.transformer.layers.", encLayers)
		markLayer(name, "mimi.decoder_transformer.transformer.layers.", decLayers)
	}
	inv.FlowLayers, inv.FlowBlocks = len(flowLayers), len(flowBlocks)
	inv.MimiEncoderLayers, inv.MimiDecoderLayers = len(encLayers), len(decLayers)
	required := expectedTensorShapes(cfg)
	for name, shape := range required {
		info, ok := infos[name]
		if !ok {
			inv.Missing = append(inv.Missing, name)
			continue
		}
		if !equalShape(info.Shape, shape) {
			inv.ShapesValid = false
			inv.Issues = append(inv.Issues, fmt.Sprintf("%s shape=%v want=%v", name, info.Shape, shape))
		}
	}
	if inv.FlowLayers != cfg.FlowLM.Transformer.NumLayers {
		inv.ShapesValid = false
		inv.Issues = append(inv.Issues, fmt.Sprintf("FlowLM layers=%d want=%d", inv.FlowLayers, cfg.FlowLM.Transformer.NumLayers))
	}
	if inv.FlowBlocks != cfg.FlowLM.Flow.Depth {
		inv.ShapesValid = false
		inv.Issues = append(inv.Issues, fmt.Sprintf("flow blocks=%d want=%d", inv.FlowBlocks, cfg.FlowLM.Flow.Depth))
	}
	if inv.MimiEncoderLayers != cfg.Mimi.Transformer.NumLayers || inv.MimiDecoderLayers != cfg.Mimi.Transformer.NumLayers {
		inv.ShapesValid = false
		inv.Issues = append(inv.Issues, fmt.Sprintf("Mimi transformer layers encoder=%d decoder=%d want=%d", inv.MimiEncoderLayers, inv.MimiDecoderLayers, cfg.Mimi.Transformer.NumLayers))
	}
	for dtype := range inv.DTypes {
		if dtype != "BF16" && dtype != "F32" {
			inv.ShapesValid = false
			inv.Issues = append(inv.Issues, "unsupported dtype "+dtype)
		}
	}
	sort.Strings(inv.Missing)
	sort.Strings(inv.Issues)
	return inv
}

func (i TensorInventory) CheckpointReady() bool {
	return i.Total > 0 && i.FlowLM > 0 && i.Mimi > 0 && i.Other == 0 && len(i.Missing) == 0 && i.ShapesValid
}

func markLayer(name, prefix string, dst map[int]bool) {
	if !strings.HasPrefix(name, prefix) {
		return
	}
	rest := strings.TrimPrefix(name, prefix)
	part, _, _ := strings.Cut(rest, ".")
	if index, err := strconv.Atoi(part); err == nil {
		dst[index] = true
	}
}

func expectedTensorShapes(c Config) map[string][]int {
	h, ff, heads := c.FlowLM.Transformer.DModel, c.FlowLM.Transformer.DModel*c.FlowLM.Transformer.HiddenScale, c.FlowLM.Transformer.NumHeads
	_ = heads
	f, l := c.FlowLM.Flow.Dim, c.Mimi.InnerDim
	m, mff := c.Mimi.Transformer.DModel, c.Mimi.Transformer.DimFeedforward
	out := map[string][]int{
		"flow_lm.bos_emb":                            {l},
		"flow_lm.bos_before_voice":                   {1, 1, h},
		"flow_lm.conditioner.embed.weight":           {c.FlowLM.LookupTable.NBins + 1, h},
		"flow_lm.emb_mean":                           {l},
		"flow_lm.emb_std":                            {l},
		"flow_lm.input_linear.weight":                {h, l},
		"flow_lm.out_norm.weight":                    {h},
		"flow_lm.out_norm.bias":                      {h},
		"flow_lm.out_eos.weight":                     {1, h},
		"flow_lm.out_eos.bias":                       {1},
		"flow_lm.speaker_proj_weight":                {h, l},
		"flow_lm.flow_net.input_proj.weight":         {f, l},
		"flow_lm.flow_net.input_proj.bias":           {f},
		"flow_lm.flow_net.cond_embed.weight":         {f, h},
		"flow_lm.flow_net.cond_embed.bias":           {f},
		"flow_lm.flow_net.final_layer.linear.weight": {l, f},
		"flow_lm.flow_net.final_layer.linear.bias":   {l},
		"mimi.quantizer.output_proj.weight":          {c.Mimi.OuterDim, l, 1},
		"mimi.downsample.conv.conv.weight":           {l, c.Mimi.SEANet.Dimension, 32},
		"mimi.upsample.convtr.convtr.weight":         {c.Mimi.SEANet.Dimension, 1, 32},
	}
	for i := 0; i < c.FlowLM.Transformer.NumLayers; i++ {
		p := fmt.Sprintf("flow_lm.transformer.layers.%d.", i)
		out[p+"self_attn.in_proj.weight"] = []int{3 * h, h}
		out[p+"self_attn.out_proj.weight"] = []int{h, h}
		out[p+"linear1.weight"] = []int{ff, h}
		out[p+"linear2.weight"] = []int{h, ff}
		out[p+"norm1.weight"], out[p+"norm1.bias"] = []int{h}, []int{h}
		out[p+"norm2.weight"], out[p+"norm2.bias"] = []int{h}, []int{h}
	}
	for i := 0; i < c.FlowLM.Flow.Depth; i++ {
		p := fmt.Sprintf("flow_lm.flow_net.res_blocks.%d.", i)
		out[p+"in_ln.weight"], out[p+"in_ln.bias"] = []int{f}, []int{f}
		out[p+"mlp.0.weight"], out[p+"mlp.0.bias"] = []int{f, f}, []int{f}
		out[p+"mlp.2.weight"], out[p+"mlp.2.bias"] = []int{f, f}, []int{f}
		out[p+"adaLN_modulation.1.weight"], out[p+"adaLN_modulation.1.bias"] = []int{3 * f, f}, []int{3 * f}
	}
	for _, side := range []string{"encoder", "decoder"} {
		for i := 0; i < c.Mimi.Transformer.NumLayers; i++ {
			p := fmt.Sprintf("mimi.%s_transformer.transformer.layers.%d.", side, i)
			out[p+"self_attn.in_proj.weight"] = []int{3 * m, m}
			out[p+"self_attn.out_proj.weight"] = []int{m, m}
			out[p+"linear1.weight"] = []int{mff, m}
			out[p+"linear2.weight"] = []int{m, mff}
			out[p+"norm1.weight"], out[p+"norm1.bias"] = []int{m}, []int{m}
			out[p+"norm2.weight"], out[p+"norm2.bias"] = []int{m}, []int{m}
			out[p+"layer_scale_1.scale"], out[p+"layer_scale_2.scale"] = []int{m}, []int{m}
		}
	}
	return out
}

func equalShape(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
