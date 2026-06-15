package diffusiongemma

import (
	"fmt"
	"log"
	"time"

	"github.com/rcarmo/go-pherence/loader/gguf"
)

// ggufGlobalToSafetensors maps non-layer GGUF tensor names to their
// safetensors equivalents.
var ggufGlobalToSafetensors = map[string]string{
	"token_embd.weight":         "model.decoder.embed_tokens.weight",
	"output_norm.weight":        "model.decoder.norm.weight",
	"rope_freqs.weight":         "model.decoder.rope_freqs.weight",
	"self_cond_pre_norm.weight": "model.decoder.self_conditioning.pre_norm.weight",
	"self_cond_gate.weight":     "model.decoder.self_conditioning.gate_proj.weight",
	"self_cond_up.weight":       "model.decoder.self_conditioning.up_proj.weight",
	"self_cond_down.weight":     "model.decoder.self_conditioning.down_proj.weight",
}

// ggufLayerSuffixToSafetensors maps GGUF per-layer suffixes to safetensors
// equivalents. The GGUF name is "blk.{L}.{suffix}", while the safetensors
// name is "model.decoder.layers.{L}.{safetensors_suffix}".
var ggufLayerSuffixToSafetensors = map[string]string{
	// Attention norms and projections
	"attn_norm.weight":           "input_layernorm.weight",
	"post_attention_norm.weight": "post_attention_layernorm.weight",
	"attn_q.weight":              "self_attn.q_proj.weight",
	"attn_k.weight":              "self_attn.k_proj.weight",
	"attn_v.weight":              "self_attn.v_proj.weight",
	"attn_output.weight":         "self_attn.o_proj.weight",
	"attn_q_norm.weight":         "self_attn.q_norm.weight",
	"attn_k_norm.weight":         "self_attn.k_norm.weight",

	// Dense MLP
	"ffn_norm.weight": "pre_feedforward_layernorm.weight",
	"ffn_gate.weight": "mlp.gate_proj.weight",
	"ffn_up.weight":   "mlp.up_proj.weight",
	"ffn_down.weight": "mlp.down_proj.weight",

	// Layer norms
	"post_ffw_norm.weight":   "post_feedforward_layernorm.weight",
	"post_ffw_norm_1.weight": "post_feedforward_layernorm_1.weight",
	"post_ffw_norm_2.weight": "post_feedforward_layernorm_2.weight",
	"pre_ffw_norm_2.weight":  "pre_feedforward_layernorm_2.weight",

	// Layer scalar
	"layer_output_scale.weight":     "layer_scalar",
	"enc_layer_output_scale.weight": "enc_layer_scalar",

	// Router
	"ffn_gate_inp.weight": "router.proj.weight",
	"ffn_gate_inp.scale":  "router.scale",
	"ffn_down_exps.scale": "router.per_expert_scale",

	// Expert MoE (3D tensors — handled by GGUFExpertIndex, not CachedFloatTensor)
	"ffn_gate_up_exps.weight": "experts.gate_up_proj",
	"ffn_down_exps.weight":    "experts.down_proj",
}

// ggufShapeToPyTorch converts GGUF's innermost-first shape to PyTorch's
// outermost-first convention. For 2D: GGUF [inDim, outDim] → PT [outDim, inDim].
func ggufShapeToPyTorch(ggufShape []uint64) []int {
	out := make([]int, len(ggufShape))
	for i, d := range ggufShape {
		out[i] = int(d)
	}
	// Reverse for 2D+ tensors
	if len(out) >= 2 {
		for i, j := 0, len(out)-1; i < j; i, j = i+1, j-1 {
			out[i], out[j] = out[j], out[i]
		}
	}
	return out
}

// OpenTextWeightsFromGGUF creates a TextWeights backed by a GGUF file,
// using exactly the same Q4_K_M weights as llama.cpp. All 1D/2D tensors are
// dequantized to F32 eagerly; expert 3D tensors are handled by GGUFExpertIndex.
//
// The GGUF file must remain open (mmap'd) for the lifetime of the weights.
func OpenTextWeightsFromGGUF(g *gguf.GGUF, shape Shape) (*TextWeights, error) {
	t0 := time.Now()

	w := &TextWeights{
		Plan:       TextTensorPlan{Ready: true},
		shards:     nil,
		floatCache: make(map[string]FloatTensor),
		noEvict:    true, // GGUF: all weights pre-cached, cannot reload
	}

	// Helper: resolve a GGUF tensor, create binding, and dequant to cache.
	resolve := func(ggufName, stName string) *TensorBinding {
		ti, ok := g.TensorByName(ggufName)
		if !ok {
			return nil
		}
		ptShape := ggufShapeToPyTorch(ti.Shape)
		b := &TensorBinding{
			TensorHandle: TensorHandle{
				Name:     stName,
				Group:    ClassifyTensorName(stName),
				Required: true,
			},
			DType: "F32",
			Shape: ptShape,
		}
		// Skip 3D expert tensors (handled by GGUFExpertIndex)
		if len(ti.Shape) == 3 {
			return b
		}
		// Dequant and cache
		raw, err := g.Raw(ti)
		if err != nil {
			log.Printf("GGUF→TextWeights: skip %s: %v", ggufName, err)
			return b
		}
		n := 1
		for _, d := range ti.Shape {
			n *= int(d)
		}
		f32, err := gguf.DequantToF32(raw, ti.QType, n)
		if err != nil {
			log.Printf("GGUF→TextWeights: skip %s (%s): %v", ggufName, ti.QType, err)
			return b
		}
		w.floatCache[stName] = FloatTensor{Data: f32, Shape: ptShape, DType: "F32"}
		return b
	}

	// Keep the original quantized tied token embedding as the GGUF LM-head
	// source. The F32 cache is still populated for prompt/canvas embeddings and
	// self-conditioning, but LM-head can use the Q6_K rows directly.
	if ti, ok := g.TensorByName("token_embd.weight"); ok && len(ti.Shape) == 2 {
		if qm, err := g.MatrixFromTensor(ti); err == nil {
			w.ggufTokenEmbd = qm
		} else {
			log.Printf("GGUF→TextWeights: token_embd quant matrix unavailable: %v", err)
		}
	}

	// Global tensors
	for ggufName, stName := range ggufGlobalToSafetensors {
		b := resolve(ggufName, stName)
		if b == nil {
			w.Plan.Ready = false
			w.Plan.Missing = append(w.Plan.Missing, stName+" (gguf: "+ggufName+")")
			continue
		}
		w.Plan.Globals = append(w.Plan.Globals, b.TensorHandle)
		w.Globals = append(w.Globals, *b)
	}

	// Per-layer tensors
	for layer := 0; layer < shape.TextLayers; layer++ {
		lt := layerTypeAt(shape.LayerTypes, layer)
		lp := LayerTensorPlan{Layer: layer, Type: lt}
		lw := LayerWeights{Layer: layer, Type: lt}

		for ggufSuffix, stSuffix := range ggufLayerSuffixToSafetensors {
			ggufName := fmt.Sprintf("blk.%d.%s", layer, ggufSuffix)
			stName := fmt.Sprintf("model.decoder.layers.%d.%s", layer, stSuffix)
			b := resolve(ggufName, stName)
			if b == nil {
				continue
			}
			lp.Handles = append(lp.Handles, b.TensorHandle)
			lw.Bindings = append(lw.Bindings, *b)
		}
		w.Plan.Layers = append(w.Plan.Layers, lp)
		w.Layers = append(w.Layers, lw)
	}

	log.Printf("GGUF→TextWeights: %d tensors dequantized to F32 in %s",
		len(w.floatCache), time.Since(t0).Round(time.Millisecond))

	return w, nil
}
