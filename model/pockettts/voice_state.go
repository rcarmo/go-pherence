package pockettts

import (
	"fmt"
	"math"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

const EnglishVoiceRevision = "e81d79e8194ad4c7ce879c87a4258ef20cbf2487"

// LoadVoiceState imports the released flattened safetensors state. It accepts
// only synchronized six-layer linear K/V caches and copies values before close.
func LoadVoiceState(path string, model *TransformerCPU, extraCapacity int) (*TransformerState, error) {
	if model == nil || extraCapacity < 0 {
		return nil, fmt.Errorf("invalid Pocket TTS voice-state request")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	infos := file.TensorInfos()
	if len(infos) != 2*len(model.Layers) {
		return nil, fmt.Errorf("Pocket TTS voice state tensors=%d want=%d", len(infos), 2*len(model.Layers))
	}
	var out *TransformerState
	for layer := range model.Layers {
		prefix := fmt.Sprintf("transformer.layers.%d.self_attn", layer)
		offset, shape, err := file.GetFloat32(prefix + "/offset")
		if err != nil {
			return nil, err
		}
		if !equalShape(shape, []int{1}) || len(offset) != 1 || offset[0] <= 0 || offset[0] != float32(int(offset[0])) {
			return nil, fmt.Errorf("invalid Pocket TTS voice offset layer=%d shape=%v value=%v", layer, shape, offset)
		}
		length := int(offset[0])
		if out == nil {
			out, err = model.NewState(length + extraCapacity)
			if err != nil {
				return nil, err
			}
			out.Position = length
		} else if out.Position != length {
			return nil, fmt.Errorf("Pocket TTS voice offsets disagree layer=%d got=%d want=%d", layer, length, out.Position)
		}
		cache, shape, err := file.GetFloat32(prefix + "/cache")
		if err != nil {
			return nil, err
		}
		wantShape := []int{2, 1, length, model.Heads, model.HeadDim}
		if !equalShape(shape, wantShape) || len(cache) != 2*length*model.Width || !finiteF32(cache) {
			return nil, fmt.Errorf("invalid Pocket TTS voice cache layer=%d shape=%v want=%v", layer, shape, wantShape)
		}
		state := &out.Layers[layer]
		copy(state.Keys[:length*model.Width], cache[:length*model.Width])
		copy(state.Values[:length*model.Width], cache[length*model.Width:])
		state.Length = length
	}
	return out, nil
}

func (m *FlowLMCPU) PromptText(state *TransformerState, tokenIDs []uint32) error {
	if m == nil || state == nil || len(tokenIDs) == 0 {
		return fmt.Errorf("invalid Pocket TTS text prompt")
	}
	h := m.Config.FlowLM.Transformer.DModel
	for _, id := range tokenIDs {
		if int(id) >= m.Config.FlowLM.LookupTable.NBins {
			return fmt.Errorf("Pocket TTS token %d out of range", id)
		}
		if err := m.Transformer.StepInto(state.HiddenA, m.Embedding[int(id)*h:(int(id)+1)*h], state); err != nil {
			return err
		}
	}
	return nil
}

// FirstLatent advances BOS through the prompted state, applies the pinned noise
// sample to the flow head, and returns normalized latent plus EOS decision.
func (m *FlowLMCPU) FirstLatent(state *TransformerState, noise []float32, flow *FlowHeadCPU, steps int, eosThreshold float32) ([]float32, bool, error) {
	if m == nil || state == nil || flow == nil || len(m.BOS) != m.Config.Mimi.InnerDim || len(noise) != m.Config.Mimi.InnerDim || math.IsNaN(float64(eosThreshold)) {
		return nil, false, fmt.Errorf("invalid Pocket TTS first-latent input")
	}
	row := make([]float32, m.Config.FlowLM.Transformer.DModel)
	if err := m.Input.Forward(row, m.BOS); err != nil {
		return nil, false, err
	}
	hidden := make([]float32, m.Config.FlowLM.Transformer.DModel)
	if err := m.Transformer.StepInto(hidden, row, state); err != nil {
		return nil, false, err
	}
	eos := make([]float32, 1)
	if err := m.EOS.Forward(eos, hidden); err != nil {
		return nil, false, err
	}
	out, velocity := make([]float32, len(noise)), make([]float32, len(noise))
	network := func(dst []float32, start, target float32, current []float32) error {
		return flow.Forward(dst, hidden, []float32{start, target}, current)
	}
	if err := LSDDecodeSIMD(out, noise, steps, velocity, network); err != nil {
		return nil, false, err
	}
	return out, eos[0] > eosThreshold, nil
}
