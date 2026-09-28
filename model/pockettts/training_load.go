package pockettts

import (
	"fmt"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func materializeOwnedLinearF32(src *safetensors.File, prefix string, linear *LinearF32) error {
	if err := materializeLinearF32(src, prefix, linear); err != nil {
		return err
	}
	linear.WeightBF16 = nil
	return nil
}

func releasedTrainingTopology(cfg Config) error {
	flow, transformer := cfg.FlowLM.Flow, cfg.FlowLM.Transformer
	if flow.Type != "lsd" || flow.Depth != 6 || flow.Dim != 512 || transformer.DModel != 1024 || transformer.NumHeads != 16 || transformer.NumLayers != 6 || transformer.HiddenScale != 4 || transformer.DimFeedforward != 0 || transformer.LayerScale != 0 || transformer.Context != 0 || transformer.MaxPeriod != 10000 || cfg.FlowLM.LookupTable.NBins != 4000 || cfg.Mimi.InnerDim != 32 {
		return fmt.Errorf("unsupported Pocket TTS released training topology")
	}
	return nil
}

// LoadFlowLMTrainingCPU loads the released FlowLM checkpoint into mutable,
// owned F32 storage. It includes the voice-conditioning parameters and latent
// statistics omitted by the preset-voice inference boundary.
func LoadFlowLMTrainingCPU(src *safetensors.File, cfg Config) (*FlowLMTrainingCPU, error) {
	if src == nil {
		return nil, fmt.Errorf("nil Pocket TTS training tensor source")
	}
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if err := releasedTrainingTopology(cfg); err != nil {
		return nil, err
	}
	inference, err := LoadFlowLMCPU(src, cfg)
	if err != nil {
		return nil, err
	}
	hidden, latent := cfg.FlowLM.Transformer.DModel, cfg.Mimi.InnerDim
	beforeVoice, shape, err := src.GetFloat32("flow_lm.bos_before_voice")
	if err != nil {
		return nil, err
	}
	if !equalShape(shape, []int{1, 1, hidden}) || len(beforeVoice) != hidden || !finiteF32(beforeVoice) {
		return nil, fmt.Errorf("invalid Pocket TTS flow_lm.bos_before_voice")
	}
	speaker, shape, err := src.GetFloat32("flow_lm.speaker_proj_weight")
	if err != nil {
		return nil, err
	}
	if !equalShape(shape, []int{hidden, latent}) || len(speaker) != hidden*latent || !finiteF32(speaker) {
		return nil, fmt.Errorf("invalid Pocket TTS flow_lm.speaker_proj_weight")
	}
	mean, err := loadVectorF32(src, "flow_lm.emb_mean", latent)
	if err != nil {
		return nil, err
	}
	std, err := loadVectorF32(src, "flow_lm.emb_std", latent)
	if err != nil {
		return nil, err
	}
	if err = materializeOwnedLinearF32(src, "flow_lm.input_linear", &inference.Input); err != nil {
		return nil, err
	}
	if err = materializeOwnedLinearF32(src, "flow_lm.out_eos", &inference.EOS); err != nil {
		return nil, err
	}
	for i := range inference.Transformer.Layers {
		prefix := fmt.Sprintf("flow_lm.transformer.layers.%d", i)
		layer := &inference.Transformer.Layers[i]
		for _, item := range []struct {
			suffix string
			linear *LinearF32
		}{{"self_attn.in_proj", &layer.InProjection}, {"self_attn.out_proj", &layer.OutProjection}, {"linear1", &layer.FC1}, {"linear2", &layer.FC2}} {
			if err = materializeOwnedLinearF32(src, prefix+"."+item.suffix, item.linear); err != nil {
				return nil, err
			}
		}
	}
	return &FlowLMTrainingCPU{
		Embedding: inference.Embedding, Vocabulary: cfg.FlowLM.LookupTable.NBins + 1,
		Hidden: hidden, LatentDim: latent, BOS: inference.BOS,
		BOSBeforeVoice: beforeVoice, LatentMean: mean, LatentStd: std,
		SpeakerProjection: LinearF32{Weight: speaker, In: latent, Out: hidden},
		Input:             inference.Input, Transformer: inference.Transformer, EOS: inference.EOS,
	}, nil
}

// LoadFlowHeadTrainingCPU loads the released flow head into mutable, owned F32
// storage suitable for exact native backward and optimiser updates.
func LoadFlowHeadTrainingCPU(src *safetensors.File, cfg Config) (*FlowHeadCPU, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if err := releasedTrainingTopology(cfg); err != nil {
		return nil, err
	}
	flow, err := LoadFlowHeadCPU(src, cfg)
	if err != nil {
		return nil, err
	}
	items := []struct {
		prefix string
		linear *LinearF32
	}{{"flow_lm.flow_net.input_proj", &flow.Input}, {"flow_lm.flow_net.cond_embed", &flow.Condition}}
	for i := range flow.Time {
		prefix := fmt.Sprintf("flow_lm.flow_net.time_embed.%d", i)
		items = append(items, struct {
			prefix string
			linear *LinearF32
		}{prefix + ".mlp.0", &flow.Time[i].FC1}, struct {
			prefix string
			linear *LinearF32
		}{prefix + ".mlp.2", &flow.Time[i].FC2})
	}
	for i := range flow.Blocks {
		prefix := fmt.Sprintf("flow_lm.flow_net.res_blocks.%d", i)
		items = append(items, struct {
			prefix string
			linear *LinearF32
		}{prefix + ".mlp.0", &flow.Blocks[i].FC1}, struct {
			prefix string
			linear *LinearF32
		}{prefix + ".mlp.2", &flow.Blocks[i].FC2}, struct {
			prefix string
			linear *LinearF32
		}{prefix + ".adaLN_modulation.1", &flow.Blocks[i].Modulation})
	}
	items = append(items, struct {
		prefix string
		linear *LinearF32
	}{"flow_lm.flow_net.final_layer.adaLN_modulation.1", &flow.Final.Modulation}, struct {
		prefix string
		linear *LinearF32
	}{"flow_lm.flow_net.final_layer.linear", &flow.Final.Linear})
	for _, item := range items {
		if err = materializeOwnedLinearF32(src, item.prefix, item.linear); err != nil {
			return nil, err
		}
	}
	return flow, nil
}

// NewDefaultLSDWeightMLP returns upstream's normalized-LSD w_s_t topology:
// 2→32→32→32→1, with every affine weight and bias zero-initialised.
func NewDefaultLSDWeightMLP() *LSDWeightMLP {
	dims := [][2]int{{2, 32}, {32, 32}, {32, 32}, {32, 1}}
	out := &LSDWeightMLP{Layers: make([]LinearF32, len(dims))}
	for i, dim := range dims {
		out.Layers[i] = LinearF32{Weight: make([]float32, dim[0]*dim[1]), Bias: make([]float32, dim[1]), In: dim[0], Out: dim[1]}
	}
	return out
}
