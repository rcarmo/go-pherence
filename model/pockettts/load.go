package pockettts

import (
	"fmt"
	"math"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func loadLinearF32(src *safetensors.File, prefix string, in, out int, bias bool) (LinearF32, error) {
	weight, shape, err := src.GetBF16(prefix + ".weight")
	if err != nil {
		return LinearF32{}, err
	}
	if !equalShape(shape, []int{out, in}) || len(weight) != in*out {
		return LinearF32{}, fmt.Errorf("Pocket TTS %s.weight shape=%v values=%d", prefix, shape, len(weight))
	}
	linear := LinearF32{WeightBF16: weight, In: in, Out: out}
	if bias {
		linear.Bias, shape, err = src.GetFloat32(prefix + ".bias")
		if err != nil {
			return LinearF32{}, err
		}
		if !equalShape(shape, []int{out}) || len(linear.Bias) != out || !finiteF32(linear.Bias) {
			return LinearF32{}, fmt.Errorf("Pocket TTS %s.bias shape=%v values=%d", prefix, shape, len(linear.Bias))
		}
	}
	return linear, nil
}

func materializeLinearF32(src *safetensors.File, prefix string, l *LinearF32) error {
	if l == nil {
		return fmt.Errorf("nil Pocket TTS linear")
	}
	w, shape, err := src.GetFloat32(prefix + ".weight")
	if err != nil {
		return err
	}
	if !equalShape(shape, []int{l.Out, l.In}) || len(w) != l.Out*l.In || !finiteF32(w) {
		return fmt.Errorf("invalid Pocket TTS materialized linear %s", prefix)
	}
	l.Weight = w
	return nil
}

func loadVectorF32(src *safetensors.File, name string, size int) ([]float32, error) {
	values, shape, err := src.GetFloat32(name)
	if err != nil {
		return nil, err
	}
	if !equalShape(shape, []int{size}) || len(values) != size || !finiteF32(values) {
		return nil, fmt.Errorf("Pocket TTS %s shape=%v values=%d", name, shape, len(values))
	}
	return values, nil
}

func LoadTransformerCPU(src *safetensors.File, prefix string, cfg TransformerConfig, finalNorm string) (*TransformerCPU, error) {
	if src == nil {
		return nil, fmt.Errorf("nil Pocket TTS tensor source")
	}
	width, ff := cfg.DModel, cfg.DimFeedforward
	if ff == 0 {
		ff = width * cfg.HiddenScale
	}
	m := &TransformerCPU{Width: width, Heads: cfg.NumHeads, HeadDim: width / cfg.NumHeads, Context: cfg.Context, MaxPeriod: cfg.MaxPeriod, Layers: make([]TransformerLayerCPU, cfg.NumLayers)}
	if m.MaxPeriod == 0 {
		m.MaxPeriod = 10000
	}
	var err error
	for i := range m.Layers {
		p := fmt.Sprintf("%s.layers.%d", prefix, i)
		l := &m.Layers[i]
		if l.Norm1Weight, err = loadVectorF32(src, p+".norm1.weight", width); err != nil {
			return nil, err
		}
		if l.Norm1Bias, err = loadVectorF32(src, p+".norm1.bias", width); err != nil {
			return nil, err
		}
		if l.Norm2Weight, err = loadVectorF32(src, p+".norm2.weight", width); err != nil {
			return nil, err
		}
		if l.Norm2Bias, err = loadVectorF32(src, p+".norm2.bias", width); err != nil {
			return nil, err
		}
		if l.InProjection, err = loadLinearF32(src, p+".self_attn.in_proj", width, 3*width, false); err != nil {
			return nil, err
		}
		if l.OutProjection, err = loadLinearF32(src, p+".self_attn.out_proj", width, width, false); err != nil {
			return nil, err
		}
		if l.FC1, err = loadLinearF32(src, p+".linear1", width, ff, false); err != nil {
			return nil, err
		}
		if l.FC2, err = loadLinearF32(src, p+".linear2", ff, width, false); err != nil {
			return nil, err
		}
		if cfg.LayerScale != 0 {
			if l.LayerScale1, err = loadVectorF32(src, p+".layer_scale_1.scale", width); err != nil {
				return nil, err
			}
			if l.LayerScale2, err = loadVectorF32(src, p+".layer_scale_2.scale", width); err != nil {
				return nil, err
			}
		}
	}
	if finalNorm != "" {
		if m.FinalWeight, err = loadVectorF32(src, finalNorm+".weight", width); err != nil {
			return nil, err
		}
		if m.FinalBias, err = loadVectorF32(src, finalNorm+".bias", width); err != nil {
			return nil, err
		}
	}
	return m, nil
}

func LoadFlowHeadCPU(src *safetensors.File, cfg Config) (*FlowHeadCPU, error) {
	if src == nil {
		return nil, fmt.Errorf("nil Pocket TTS tensor source")
	}
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	f, h, latent := cfg.FlowLM.Flow.Dim, cfg.FlowLM.Transformer.DModel, cfg.Mimi.InnerDim
	m := &FlowHeadCPU{Blocks: make([]AdaLNResidual, cfg.FlowLM.Flow.Depth), Time: make([]TimestepMLP, 2)}
	var err error
	if m.Input, err = loadLinearF32(src, "flow_lm.flow_net.input_proj", latent, f, true); err != nil {
		return nil, err
	}
	if m.Condition, err = loadLinearF32(src, "flow_lm.flow_net.cond_embed", h, f, true); err != nil {
		return nil, err
	}
	for i := range m.Time {
		p := fmt.Sprintf("flow_lm.flow_net.time_embed.%d", i)
		if m.Time[i].Frequencies, err = loadVectorF32(src, p+".freqs", 128); err != nil {
			return nil, err
		}
		if m.Time[i].FC1, err = loadLinearF32(src, p+".mlp.0", 256, f, true); err != nil {
			return nil, err
		}
		if m.Time[i].FC2, err = loadLinearF32(src, p+".mlp.2", f, f, true); err != nil {
			return nil, err
		}
		if m.Time[i].RMSWeight, err = loadVectorF32(src, p+".mlp.3.alpha", f); err != nil {
			return nil, err
		}
		m.Time[i].RMSEpsilon = 1e-5
	}
	for i := range m.Blocks {
		p := fmt.Sprintf("flow_lm.flow_net.res_blocks.%d", i)
		b := &m.Blocks[i]
		if b.NormWeight, err = loadVectorF32(src, p+".in_ln.weight", f); err != nil {
			return nil, err
		}
		if b.NormBias, err = loadVectorF32(src, p+".in_ln.bias", f); err != nil {
			return nil, err
		}
		if b.FC1, err = loadLinearF32(src, p+".mlp.0", f, f, true); err != nil {
			return nil, err
		}
		if b.FC2, err = loadLinearF32(src, p+".mlp.2", f, f, true); err != nil {
			return nil, err
		}
		if b.Modulation, err = loadLinearF32(src, p+".adaLN_modulation.1", f, 3*f, true); err != nil {
			return nil, err
		}
		b.Epsilon = 1e-6
	}
	if m.Final.Modulation, err = loadLinearF32(src, "flow_lm.flow_net.final_layer.adaLN_modulation.1", f, 2*f, true); err != nil {
		return nil, err
	}
	if m.Final.Linear, err = loadLinearF32(src, "flow_lm.flow_net.final_layer.linear", f, latent, true); err != nil {
		return nil, err
	}
	m.Final.Epsilon = 1e-6
	return m, nil
}

func finiteF32(values []float32) bool {
	for _, value := range values {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return false
		}
	}
	return true
}
