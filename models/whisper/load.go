package whisper

import (
	"fmt"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

// LoadModel loads a Whisper model from a safetensors file.
// The weight names follow HuggingFace Whisper convention:
//
//	model.encoder.conv1.weight  [d_model, mel_bins, 3]
//	model.encoder.conv1.bias    [d_model]
//	model.encoder.conv2.weight  [d_model, d_model, 3]
//	model.encoder.conv2.bias    [d_model]
//	model.encoder.embed_positions.weight [max_source_positions, d_model]
//	model.encoder.layers.{i}.self_attn_layer_norm.weight
//	model.encoder.layers.{i}.self_attn_layer_norm.bias
//	model.encoder.layers.{i}.self_attn.q_proj.weight
//	model.encoder.layers.{i}.self_attn.q_proj.bias
//	model.encoder.layers.{i}.self_attn.k_proj.weight
//	model.encoder.layers.{i}.self_attn.v_proj.weight
//	model.encoder.layers.{i}.self_attn.v_proj.bias
//	model.encoder.layers.{i}.self_attn.out_proj.weight
//	model.encoder.layers.{i}.self_attn.out_proj.bias
//	model.encoder.layers.{i}.final_layer_norm.weight
//	model.encoder.layers.{i}.final_layer_norm.bias
//	model.encoder.layers.{i}.fc1.weight
//	model.encoder.layers.{i}.fc1.bias
//	model.encoder.layers.{i}.fc2.weight
//	model.encoder.layers.{i}.fc2.bias
//	model.decoder.embed_tokens.weight
//	model.decoder.embed_positions.weight
//	model.decoder.layers.{i}.self_attn_layer_norm.weight/bias
//	model.decoder.layers.{i}.self_attn.q_proj.weight/bias
//	model.decoder.layers.{i}.self_attn.k_proj.weight/bias
//	model.decoder.layers.{i}.self_attn.v_proj.weight/bias
//	model.decoder.layers.{i}.self_attn.out_proj.weight/bias
//	model.decoder.layers.{i}.encoder_attn_layer_norm.weight/bias
//	model.decoder.layers.{i}.encoder_attn.q_proj.weight/bias
//	model.decoder.layers.{i}.encoder_attn.k_proj.weight/bias
//	model.decoder.layers.{i}.encoder_attn.v_proj.weight/bias
//	model.decoder.layers.{i}.encoder_attn.out_proj.weight/bias
//	model.decoder.layers.{i}.final_layer_norm.weight/bias
//	model.decoder.layers.{i}.fc1.weight/bias
//	model.decoder.layers.{i}.fc2.weight/bias
//	model.decoder.layer_norm.weight/bias
func LoadModel(path string, cfg Config) (*Encoder, *Decoder, error) {
	f, err := safetensors.Open(path)
	if err != nil {
		return nil, nil, fmt.Errorf("open safetensors: %w", err)
	}

	get := func(name string) ([]float32, error) {
		data, _, err := f.GetFloat32(name)
		return data, err
	}

	// getLinear loads a 2D weight matrix and transposes it from [out, in] to [in, out]
	// HuggingFace stores nn.Linear weights as [out_features, in_features] but our
	// linearForwardOpt expects the transposed layout for correct matrix-vector multiply.
	getLinear := func(name string, rows, cols int) ([]float32, error) {
		data, _, err := f.GetFloat32(name)
		if err != nil || len(data) != rows*cols {
			return data, err
		}
		// Transpose [rows, cols] → [cols, rows]
		t := make([]float32, rows*cols)
		for r := 0; r < rows; r++ {
			for c := 0; c < cols; c++ {
				t[c*rows+r] = data[r*cols+c]
			}
		}
		return t, nil
	}

	enc := NewEncoder(cfg)

	// Conv stem
	enc.Conv1Weight, err = get("model.encoder.conv1.weight")
	if err != nil {
		return nil, nil, fmt.Errorf("conv1.weight: %w", err)
	}
	enc.Conv1Bias, err = get("model.encoder.conv1.bias")
	if err != nil {
		return nil, nil, fmt.Errorf("conv1.bias: %w", err)
	}
	enc.Conv2Weight, err = get("model.encoder.conv2.weight")
	if err != nil {
		return nil, nil, fmt.Errorf("conv2.weight: %w", err)
	}
	enc.Conv2Bias, err = get("model.encoder.conv2.bias")
	if err != nil {
		return nil, nil, fmt.Errorf("conv2.bias: %w", err)
	}

	// Encoder positional embeddings (learned, not sinusoidal in HF Whisper)
	posEmbed, err := get("model.encoder.embed_positions.weight")
	if err == nil {
		enc.PosEmbed = posEmbed
	}

	// Final encoder LayerNorm
	enc.FinalLNWeight, _ = get("model.encoder.layer_norm.weight")
	enc.FinalLNBias, _ = get("model.encoder.layer_norm.bias")

	// Encoder layers
	for i := 0; i < cfg.EncoderLayers; i++ {
		l := &enc.Layers[i]
		prefix := fmt.Sprintf("model.encoder.layers.%d", i)

		l.AttnLNWeight, _ = get(prefix + ".self_attn_layer_norm.weight")
		l.AttnLNBias, _ = get(prefix + ".self_attn_layer_norm.bias")
		l.QWeight, _ = getLinear(prefix+".self_attn.q_proj.weight", cfg.EncoderDModel, cfg.EncoderDModel)
		l.QBias, _ = get(prefix + ".self_attn.q_proj.bias")
		l.KWeight, _ = getLinear(prefix+".self_attn.k_proj.weight", cfg.EncoderDModel, cfg.EncoderDModel)
		l.KBias, _ = get(prefix + ".self_attn.k_proj.bias") // may not exist
		l.VWeight, _ = getLinear(prefix+".self_attn.v_proj.weight", cfg.EncoderDModel, cfg.EncoderDModel)
		l.VBias, _ = get(prefix + ".self_attn.v_proj.bias")
		l.OWeight, _ = getLinear(prefix+".self_attn.out_proj.weight", cfg.EncoderDModel, cfg.EncoderDModel)
		l.OBias, _ = get(prefix + ".self_attn.out_proj.bias")

		l.MLPLNWeight, _ = get(prefix + ".final_layer_norm.weight")
		l.MLPLNBias, _ = get(prefix + ".final_layer_norm.bias")
		l.FC1Weight, _ = getLinear(prefix+".fc1.weight", cfg.EncoderFFNDim, cfg.EncoderDModel)
		l.FC1Bias, _ = get(prefix + ".fc1.bias")
		l.FC2Weight, _ = getLinear(prefix+".fc2.weight", cfg.EncoderDModel, cfg.EncoderFFNDim)
		l.FC2Bias, _ = get(prefix + ".fc2.bias")
	}

	// Decoder
	dec := NewDecoder(cfg)
	dec.TokenEmbed, _ = get("model.decoder.embed_tokens.weight")
	dec.PosEmbed, _ = get("model.decoder.embed_positions.weight")
	dec.FinalLNWeight, _ = get("model.decoder.layer_norm.weight")
	dec.FinalLNBias, _ = get("model.decoder.layer_norm.bias")

	for i := 0; i < cfg.DecoderLayers; i++ {
		l := &dec.Layers[i]
		prefix := fmt.Sprintf("model.decoder.layers.%d", i)

		l.SelfAttnLNWeight, _ = get(prefix + ".self_attn_layer_norm.weight")
		l.SelfAttnLNBias, _ = get(prefix + ".self_attn_layer_norm.bias")
		l.SelfQWeight, _ = getLinear(prefix+".self_attn.q_proj.weight", cfg.DecoderDModel, cfg.DecoderDModel)
		l.SelfQBias, _ = get(prefix + ".self_attn.q_proj.bias")
		l.SelfKWeight, _ = getLinear(prefix+".self_attn.k_proj.weight", cfg.DecoderDModel, cfg.DecoderDModel)
		l.SelfKBias, _ = get(prefix + ".self_attn.k_proj.bias")
		l.SelfVWeight, _ = getLinear(prefix+".self_attn.v_proj.weight", cfg.DecoderDModel, cfg.DecoderDModel)
		l.SelfVBias, _ = get(prefix + ".self_attn.v_proj.bias")
		l.SelfOWeight, _ = getLinear(prefix+".self_attn.out_proj.weight", cfg.DecoderDModel, cfg.DecoderDModel)
		l.SelfOBias, _ = get(prefix + ".self_attn.out_proj.bias")

		l.CrossAttnLNWeight, _ = get(prefix + ".encoder_attn_layer_norm.weight")
		l.CrossAttnLNBias, _ = get(prefix + ".encoder_attn_layer_norm.bias")
		l.CrossQWeight, _ = getLinear(prefix+".encoder_attn.q_proj.weight", cfg.DecoderDModel, cfg.DecoderDModel)
		l.CrossQBias, _ = get(prefix + ".encoder_attn.q_proj.bias")
		l.CrossKWeight, _ = getLinear(prefix+".encoder_attn.k_proj.weight", cfg.DecoderDModel, cfg.DecoderDModel)
		l.CrossKBias, _ = get(prefix + ".encoder_attn.k_proj.bias")
		l.CrossVWeight, _ = getLinear(prefix+".encoder_attn.v_proj.weight", cfg.DecoderDModel, cfg.DecoderDModel)
		l.CrossVBias, _ = get(prefix + ".encoder_attn.v_proj.bias")
		l.CrossOWeight, _ = getLinear(prefix+".encoder_attn.out_proj.weight", cfg.DecoderDModel, cfg.DecoderDModel)
		l.CrossOBias, _ = get(prefix + ".encoder_attn.out_proj.bias")

		l.MLPLNWeight, _ = get(prefix + ".final_layer_norm.weight")
		l.MLPLNBias, _ = get(prefix + ".final_layer_norm.bias")
		l.FC1Weight, _ = getLinear(prefix+".fc1.weight", cfg.EncoderFFNDim, cfg.EncoderDModel)
		l.FC1Bias, _ = get(prefix + ".fc1.bias")
		l.FC2Weight, _ = getLinear(prefix+".fc2.weight", cfg.EncoderDModel, cfg.EncoderFFNDim)
		l.FC2Bias, _ = get(prefix + ".fc2.bias")
	}

	return enc, dec, nil
}
