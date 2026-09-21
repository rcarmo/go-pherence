package whisper

import "fmt"

// Float32TensorSource is the subset of safetensors loading required by the
// encoder. BF16 checkpoint tensors are widened by the source implementation.
type Float32TensorSource interface {
	GetFloat32(name string) ([]float32, []int, error)
}

// LoadEncoderSource loads an encoder from a tensor source using the supplied
// Hugging Face prefix (for example "model.whisper_encoder"). It validates
// every tensor shape and never silently accepts missing layer parameters.
func LoadEncoderSource(source Float32TensorSource, prefix string, cfg Config) (*Encoder, error) {
	if source == nil {
		return nil, fmt.Errorf("whisper: nil encoder tensor source")
	}
	if cfg.NumMelBins <= 0 || cfg.MaxLength <= 0 || cfg.EncoderLayers <= 0 || cfg.EncoderDModel <= 0 || cfg.EncoderHeads <= 0 || cfg.EncoderFFNDim <= 0 || cfg.HeadDim*cfg.EncoderHeads != cfg.EncoderDModel {
		return nil, fmt.Errorf("whisper: invalid encoder config: %+v", cfg)
	}
	get := func(suffix string, shape ...int) ([]float32, error) {
		name := prefix + "." + suffix
		values, gotShape, err := source.GetFloat32(name)
		if err != nil {
			return nil, fmt.Errorf("whisper: load %s: %w", name, err)
		}
		if !sameShape(gotShape, shape) {
			return nil, fmt.Errorf("whisper: tensor %s shape %v, want %v", name, gotShape, shape)
		}
		return values, nil
	}

	enc := NewEncoder(cfg)
	var err error
	if enc.Conv1Weight, err = get("conv1.weight", cfg.EncoderDModel, cfg.NumMelBins, 3); err != nil {
		return nil, err
	}
	if enc.Conv1Bias, err = get("conv1.bias", cfg.EncoderDModel); err != nil {
		return nil, err
	}
	if enc.Conv2Weight, err = get("conv2.weight", cfg.EncoderDModel, cfg.EncoderDModel, 3); err != nil {
		return nil, err
	}
	if enc.Conv2Bias, err = get("conv2.bias", cfg.EncoderDModel); err != nil {
		return nil, err
	}
	if enc.PosEmbed, err = get("embed_positions.weight", cfg.MaxLength/2, cfg.EncoderDModel); err != nil {
		return nil, err
	}
	if enc.FinalLNWeight, err = get("layer_norm.weight", cfg.EncoderDModel); err != nil {
		return nil, err
	}
	if enc.FinalLNBias, err = get("layer_norm.bias", cfg.EncoderDModel); err != nil {
		return nil, err
	}

	for i := range enc.Layers {
		layer := &enc.Layers[i]
		base := fmt.Sprintf("layers.%d.", i)
		if layer.AttnLNWeight, err = get(base+"self_attn_layer_norm.weight", cfg.EncoderDModel); err != nil {
			return nil, err
		}
		if layer.AttnLNBias, err = get(base+"self_attn_layer_norm.bias", cfg.EncoderDModel); err != nil {
			return nil, err
		}
		if layer.QWeight, err = get(base+"self_attn.q_proj.weight", cfg.EncoderDModel, cfg.EncoderDModel); err != nil {
			return nil, err
		}
		if layer.QBias, err = get(base+"self_attn.q_proj.bias", cfg.EncoderDModel); err != nil {
			return nil, err
		}
		if layer.KWeight, err = get(base+"self_attn.k_proj.weight", cfg.EncoderDModel, cfg.EncoderDModel); err != nil {
			return nil, err
		}
		if layer.VWeight, err = get(base+"self_attn.v_proj.weight", cfg.EncoderDModel, cfg.EncoderDModel); err != nil {
			return nil, err
		}
		if layer.VBias, err = get(base+"self_attn.v_proj.bias", cfg.EncoderDModel); err != nil {
			return nil, err
		}
		if layer.OWeight, err = get(base+"self_attn.out_proj.weight", cfg.EncoderDModel, cfg.EncoderDModel); err != nil {
			return nil, err
		}
		if layer.OBias, err = get(base+"self_attn.out_proj.bias", cfg.EncoderDModel); err != nil {
			return nil, err
		}
		if layer.MLPLNWeight, err = get(base+"final_layer_norm.weight", cfg.EncoderDModel); err != nil {
			return nil, err
		}
		if layer.MLPLNBias, err = get(base+"final_layer_norm.bias", cfg.EncoderDModel); err != nil {
			return nil, err
		}
		if layer.FC1Weight, err = get(base+"fc1.weight", cfg.EncoderFFNDim, cfg.EncoderDModel); err != nil {
			return nil, err
		}
		if layer.FC1Bias, err = get(base+"fc1.bias", cfg.EncoderFFNDim); err != nil {
			return nil, err
		}
		if layer.FC2Weight, err = get(base+"fc2.weight", cfg.EncoderDModel, cfg.EncoderFFNDim); err != nil {
			return nil, err
		}
		if layer.FC2Bias, err = get(base+"fc2.bias", cfg.EncoderDModel); err != nil {
			return nil, err
		}
	}
	prepackA100EncoderWeights(enc)
	return enc, nil
}

func sameShape(got, want []int) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range want {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}
