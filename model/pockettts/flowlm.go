package pockettts

import (
	"fmt"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

type FlowLMCPU struct {
	Config      Config
	Embedding   []float32
	BOS         []float32
	Input       LinearF32
	Transformer *TransformerCPU
	EOS         LinearF32
}

func LoadFlowLMCPU(src *safetensors.File, cfg Config) (*FlowLMCPU, error) {
	if src == nil {
		return nil, fmt.Errorf("nil Pocket TTS tensor source")
	}
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	h, l := cfg.FlowLM.Transformer.DModel, cfg.Mimi.InnerDim
	emb, shape, err := src.GetFloat32("flow_lm.conditioner.embed.weight")
	if err != nil {
		return nil, err
	}
	if !equalShape(shape, []int{cfg.FlowLM.LookupTable.NBins + 1, h}) || !finiteF32(emb) {
		return nil, fmt.Errorf("invalid Pocket TTS embedding shape=%v", shape)
	}
	m := &FlowLMCPU{Config: cfg, Embedding: emb}
	if m.BOS, err = loadVectorF32(src, "flow_lm.bos_emb", l); err != nil {
		return nil, err
	}
	if m.Input, err = loadLinearF32(src, "flow_lm.input_linear", l, h, false); err != nil {
		return nil, err
	}
	if m.Transformer, err = LoadTransformerCPU(src, "flow_lm.transformer", cfg.FlowLM.Transformer, "flow_lm.out_norm"); err != nil {
		return nil, err
	}
	if m.EOS, err = loadLinearF32(src, "flow_lm.out_eos", h, 1, true); err != nil {
		return nil, err
	}
	return m, nil
}

// Prefill executes text embeddings followed by one latent row and returns the
// normalized final row and EOS logit. It is the stateless reference boundary;
// incremental KV state uses the same layer arithmetic in the next slice.
func (m *FlowLMCPU) Prefill(tokenIDs []uint32, latent []float32) ([]float32, float32, error) {
	if m == nil || len(tokenIDs) == 0 || len(latent) != m.Config.Mimi.InnerDim {
		return nil, 0, fmt.Errorf("invalid Pocket TTS prefill input")
	}
	h := m.Config.FlowLM.Transformer.DModel
	rows := len(tokenIDs) + 1
	sequence := make([]float32, rows*h)
	for row, id := range tokenIDs {
		if int(id) >= m.Config.FlowLM.LookupTable.NBins {
			return nil, 0, fmt.Errorf("Pocket TTS token %d out of range", id)
		}
		copy(sequence[row*h:(row+1)*h], m.Embedding[int(id)*h:(int(id)+1)*h])
	}
	if err := m.Input.Forward(sequence[(rows-1)*h:], latent); err != nil {
		return nil, 0, err
	}
	hidden, err := m.Transformer.Forward(sequence, rows)
	if err != nil {
		return nil, 0, err
	}
	last := append([]float32(nil), hidden[(rows-1)*h:]...)
	eos := make([]float32, 1)
	if err := m.EOS.Forward(eos, last); err != nil {
		return nil, 0, err
	}
	return last, eos[0], nil
}

func (m *FlowLMCPU) PrefillStreaming(tokenIDs []uint32, latent []float32) ([]float32, float32, *TransformerState, error) {
	if m == nil || len(tokenIDs) == 0 || len(latent) != m.Config.Mimi.InnerDim {
		return nil, 0, nil, fmt.Errorf("invalid Pocket TTS streaming prefill input")
	}
	h := m.Config.FlowLM.Transformer.DModel
	state, err := m.Transformer.NewState(len(tokenIDs) + 1)
	if err != nil {
		return nil, 0, nil, err
	}
	hidden := make([]float32, h)
	for _, id := range tokenIDs {
		if int(id) >= m.Config.FlowLM.LookupTable.NBins {
			return nil, 0, nil, fmt.Errorf("Pocket TTS token %d out of range", id)
		}
		if err = m.Transformer.StepInto(hidden, m.Embedding[int(id)*h:(int(id)+1)*h], state); err != nil {
			return nil, 0, nil, err
		}
	}
	row := make([]float32, h)
	if err := m.Input.Forward(row, latent); err != nil {
		return nil, 0, nil, err
	}
	if err = m.Transformer.StepInto(hidden, row, state); err != nil {
		return nil, 0, nil, err
	}
	eos := make([]float32, 1)
	if err := m.EOS.Forward(eos, hidden); err != nil {
		return nil, 0, nil, err
	}
	return hidden, eos[0], state, nil
}
