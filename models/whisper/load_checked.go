package whisper

import (
	"context"
	"fmt"
	"math"
	"sort"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

// CheckedTensorSource supplies metadata before materialising any weights.
// safetensors.File implements this interface. Byte extents share one address
// space (a sharded source must normalise offsets first). Metadata bounds and
// actual file extent validation belong to the source. It must be immutable and
// return synchronously; the loader checks context around each call, but cannot
// interrupt a GetFloat32 already running. The caller owns closing the source.
type CheckedTensorSource interface {
	Float32TensorSource
	TensorInfos() map[string]safetensors.TensorInfo
}

type speechTensorBinding struct {
	name  string
	shape []int
	dst   *[]float32
}

// speechTensorBindings is the HF Whisper layout consumed by the current Go
// model. K projections have no bias; output projection is tied to token embed.
func speechTensorBindings(w *Whisper) []speechTensorBinding {
	c, e, d := w.Config, w.Encoder, w.Decoder
	var out []speechTensorBinding
	add := func(name string, dst *[]float32, shape ...int) {
		out = append(out, speechTensorBinding{name, shape, dst})
	}
	m, ff := c.EncoderDModel, c.EncoderFFNDim
	add("model.encoder.conv1.weight", &e.Conv1Weight, m, c.NumMelBins, 3)
	add("model.encoder.conv1.bias", &e.Conv1Bias, m)
	add("model.encoder.conv2.weight", &e.Conv2Weight, m, m, 3)
	add("model.encoder.conv2.bias", &e.Conv2Bias, m)
	add("model.encoder.embed_positions.weight", &e.PosEmbed, (c.MaxLength+1)/2, m)
	add("model.encoder.layer_norm.weight", &e.FinalLNWeight, m)
	add("model.encoder.layer_norm.bias", &e.FinalLNBias, m)
	for i := range e.Layers {
		l := &e.Layers[i]
		p := fmt.Sprintf("model.encoder.layers.%d.", i)
		add(p+"self_attn_layer_norm.weight", &l.AttnLNWeight, m)
		add(p+"self_attn_layer_norm.bias", &l.AttnLNBias, m)
		add(p+"self_attn.q_proj.weight", &l.QWeight, m, m)
		add(p+"self_attn.q_proj.bias", &l.QBias, m)
		add(p+"self_attn.k_proj.weight", &l.KWeight, m, m)
		add(p+"self_attn.v_proj.weight", &l.VWeight, m, m)
		add(p+"self_attn.v_proj.bias", &l.VBias, m)
		add(p+"self_attn.out_proj.weight", &l.OWeight, m, m)
		add(p+"self_attn.out_proj.bias", &l.OBias, m)
		add(p+"final_layer_norm.weight", &l.MLPLNWeight, m)
		add(p+"final_layer_norm.bias", &l.MLPLNBias, m)
		add(p+"fc1.weight", &l.FC1Weight, ff, m)
		add(p+"fc1.bias", &l.FC1Bias, ff)
		add(p+"fc2.weight", &l.FC2Weight, m, ff)
		add(p+"fc2.bias", &l.FC2Bias, m)
	}
	m, ff = c.DecoderDModel, c.DecoderFFNDim
	add("model.decoder.embed_tokens.weight", &d.TokenEmbed, c.VocabSize, m)
	add("model.decoder.embed_positions.weight", &d.PosEmbed, c.MaxDecoderLength, m)
	add("model.decoder.layer_norm.weight", &d.FinalLNWeight, m)
	add("model.decoder.layer_norm.bias", &d.FinalLNBias, m)
	for i := range d.Layers {
		l := &d.Layers[i]
		p := fmt.Sprintf("model.decoder.layers.%d.", i)
		add(p+"self_attn_layer_norm.weight", &l.SelfAttnLNWeight, m)
		add(p+"self_attn_layer_norm.bias", &l.SelfAttnLNBias, m)
		add(p+"self_attn.q_proj.weight", &l.SelfQWeight, m, m)
		add(p+"self_attn.q_proj.bias", &l.SelfQBias, m)
		add(p+"self_attn.k_proj.weight", &l.SelfKWeight, m, m)
		add(p+"self_attn.v_proj.weight", &l.SelfVWeight, m, m)
		add(p+"self_attn.v_proj.bias", &l.SelfVBias, m)
		add(p+"self_attn.out_proj.weight", &l.SelfOWeight, m, m)
		add(p+"self_attn.out_proj.bias", &l.SelfOBias, m)
		add(p+"encoder_attn_layer_norm.weight", &l.CrossAttnLNWeight, m)
		add(p+"encoder_attn_layer_norm.bias", &l.CrossAttnLNBias, m)
		add(p+"encoder_attn.q_proj.weight", &l.CrossQWeight, m, m)
		add(p+"encoder_attn.q_proj.bias", &l.CrossQBias, m)
		add(p+"encoder_attn.k_proj.weight", &l.CrossKWeight, m, m)
		add(p+"encoder_attn.v_proj.weight", &l.CrossVWeight, m, m)
		add(p+"encoder_attn.v_proj.bias", &l.CrossVBias, m)
		add(p+"encoder_attn.out_proj.weight", &l.CrossOWeight, m, m)
		add(p+"encoder_attn.out_proj.bias", &l.CrossOBias, m)
		add(p+"final_layer_norm.weight", &l.MLPLNWeight, m)
		add(p+"final_layer_norm.bias", &l.MLPLNBias, m)
		add(p+"fc1.weight", &l.FC1Weight, ff, m)
		add(p+"fc1.bias", &l.FC1Bias, ff)
		add(p+"fc2.weight", &l.FC2Weight, m, ff)
		add(p+"fc2.bias", &l.FC2Bias, m)
	}
	return out
}

// LoadModelSourceChecked is the opt-in HF F32/F16/BF16 Whisper loader. It checks
// ALL required tensor names, dimensions, dtypes and declared byte extents before
// the first GetFloat32. Unknown tensors are rejected, except optional proj_out
// whose widened values must exactly equal the tied token embedding. It never
// tolerates missing weights or integer/quantised weights. This path does not
// support whisper.cpp GGML, quantised GEMM layouts or a vocabulary projection
// different from token embeddings.
//
// The returned model owns copies of finite widened weights and can outlive the
// source. Peak memory includes the retained model plus a transient source tensor
// and its copy. No GPU upload, global packing or inference is performed here.
// Context cancellation discards the partial model; no background workers exist.
// Config/tokenizer/generation provenance and admission to load real weights are
// the caller's responsibility. Legacy LoadModel/LoadEncoderSource are unchanged.
func LoadModelSourceChecked(ctx context.Context, source CheckedTensorSource, cfg Config) (*Whisper, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if source == nil {
		return nil, fmt.Errorf("nil checked tensor source")
	}
	if err := validatePCMConfig(cfg); err != nil {
		return nil, err
	}
	// Allocate only the small structural skeleton, not embeddings/weight buffers.
	w := &Whisper{Config: cfg, Encoder: &Encoder{cfg: cfg, Layers: make([]EncoderLayer, cfg.EncoderLayers)}, Decoder: NewDecoder(cfg)}
	bindings := speechTensorBindings(w)
	infos := source.TensorInfos()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(infos) < len(bindings) || len(infos) > len(bindings)+1 {
		return nil, fmt.Errorf("unexpected checked model tensor count")
	}
	expected := make(map[string][]int, len(bindings)+1)
	for _, b := range bindings {
		expected[b.name] = b.shape
	}
	if _, ok := infos["proj_out.weight"]; ok {
		expected["proj_out.weight"] = []int{cfg.VocabSize, cfg.DecoderDModel}
	}
	for name := range infos {
		if _, ok := expected[name]; !ok {
			return nil, fmt.Errorf("unsupported model tensor %s", name)
		}
	}
	// Sorted iteration keeps errors deterministic. Shapes derive only from the
	// previously bounded config, so multiplication below cannot overflow int.
	names := make([]string, 0, len(expected))
	for name := range expected {
		names = append(names, name)
	}
	sort.Strings(names)
	type span struct {
		start, end int
		name       string
	}
	spans := make([]span, 0, len(names))
	for _, name := range names {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		info, ok := infos[name]
		if !ok {
			return nil, fmt.Errorf("missing model tensor %s", name)
		}
		if !sameShape(info.Shape, expected[name]) {
			return nil, fmt.Errorf("tensor %s shape %v, want %v", name, info.Shape, expected[name])
		}
		bytes := 0
		switch info.DType {
		case "F32":
			bytes = 4
		case "F16", "BF16":
			bytes = 2
		default:
			return nil, fmt.Errorf("unsupported tensor %s dtype %s", name, info.DType)
		}
		elements := 1
		for _, dim := range expected[name] {
			elements *= dim
		}
		start, end := info.DataOffsets[0], info.DataOffsets[1]
		if start < 0 || end < start || end-start != elements*bytes {
			return nil, fmt.Errorf("invalid tensor %s byte extent", name)
		}
		spans = append(spans, span{start, end, name})
	}
	sort.Slice(spans, func(i, j int) bool { return spans[i].start < spans[j].start })
	for i := 1; i < len(spans); i++ {
		if spans[i].start < spans[i-1].end {
			return nil, fmt.Errorf("overlapping tensor extents: %s and %s", spans[i-1].name, spans[i].name)
		}
	}
	load := func(name string, shape []int) ([]float32, error) {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		values, got, err := source.GetFloat32(name)
		if err != nil {
			return nil, fmt.Errorf("load %s: %w", name, err)
		}
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		n := 1
		for _, dim := range shape {
			n *= dim
		}
		if !sameShape(got, shape) || len(values) != n {
			return nil, fmt.Errorf("tensor %s changed shape/length during load", name)
		}
		owned := make([]float32, n)
		for start := 0; start < n; start += 16384 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			end := min(start+16384, n)
			for _, value := range values[start:end] {
				if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
					return nil, fmt.Errorf("non-finite tensor %s", name)
				}
			}
			copy(owned[start:end], values[start:end])
		}
		return owned, nil
	}
	for _, b := range bindings {
		values, err := load(b.name, b.shape)
		if err != nil {
			return nil, err
		}
		*b.dst = values
	}
	if shape, ok := expected["proj_out.weight"]; ok {
		projection, err := load("proj_out.weight", shape)
		if err != nil {
			return nil, err
		}
		for i, value := range projection {
			if i%16384 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			if value != w.Decoder.TokenEmbed[i] {
				return nil, fmt.Errorf("untied proj_out.weight is unsupported")
			}
		}
	}
	if err := w.validatePCMModel(); err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return w, nil
}
