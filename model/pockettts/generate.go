package pockettts

import (
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

type NoiseSource func(frame int, dst []float32) error

type GeneratorCPU struct {
	FlowLM    *FlowLMCPU
	Flow      *FlowHeadCPU
	Mimi      *MimiDecoderCPU
	Mean, Std []float32
}

func NewGeneratorCPU(lm *FlowLMCPU, flow *FlowHeadCPU, mimi *MimiDecoderCPU, mean, std []float32) (*GeneratorCPU, error) {
	if lm == nil || flow == nil || mimi == nil || len(mean) != lm.Config.Mimi.InnerDim || len(std) != lm.Config.Mimi.InnerDim || !finiteF32(mean) || !finiteF32(std) {
		return nil, fmt.Errorf("invalid Pocket TTS generator")
	}
	return &GeneratorCPU{FlowLM: lm, Flow: flow, Mimi: mimi, Mean: append([]float32(nil), mean...), Std: append([]float32(nil), std...)}, nil
}

// Generate starts from an imported voice state, prompts text once, then emits
// frame-exact 24 kHz mono PCM. Noise policy is caller-owned and deterministic
// fixtures can supply exact samples. EOS countdown excludes the expiry frame,
// matching upstream generation.
func (g *GeneratorCPU) Generate(voice *TransformerState, tokens []uint32, maxFrames, framesAfterEOS, decodeSteps int, eosThreshold float32, noise NoiseSource) ([]float32, error) {
	return g.GenerateStream(voice, tokens, maxFrames, framesAfterEOS, decodeSteps, eosThreshold, noise, func([]float32) error { return nil })
}

func (g *GeneratorCPU) GenerateStream(voice *TransformerState, tokens []uint32, maxFrames, framesAfterEOS, decodeSteps int, eosThreshold float32, noise NoiseSource, emit func([]float32) error) ([]float32, error) {
	if g == nil || voice == nil || len(tokens) == 0 || maxFrames <= 0 || framesAfterEOS < 0 || decodeSteps <= 0 || noise == nil || emit == nil || math.IsNaN(float64(eosThreshold)) {
		return nil, fmt.Errorf("invalid Pocket TTS generation request")
	}
	if err := g.FlowLM.PromptText(voice, tokens); err != nil {
		return nil, err
	}
	mimiState := g.Mimi.NewStateForFrames(maxFrames)
	normalized := append([]float32(nil), g.FlowLM.BOS...)
	var pcm []float32
	eosAt := -1
	for frame := 0; frame < maxFrames; frame++ {
		row := make([]float32, g.FlowLM.Config.FlowLM.Transformer.DModel)
		if err := g.FlowLM.Input.Forward(row, normalized); err != nil {
			return nil, err
		}
		hidden := make([]float32, g.FlowLM.Config.FlowLM.Transformer.DModel)
		if err := g.FlowLM.Transformer.StepInto(hidden, row, voice); err != nil {
			return nil, err
		}
		eosValue := make([]float32, 1)
		if err := g.FlowLM.EOS.Forward(eosValue, hidden); err != nil {
			return nil, err
		}
		if eosAt < 0 && eosValue[0] > eosThreshold {
			eosAt = frame
		}
		if eosAt >= 0 && frame >= eosAt+framesAfterEOS {
			break
		}
		initial := make([]float32, len(normalized))
		if err := noise(frame, initial); err != nil {
			return nil, fmt.Errorf("Pocket TTS noise frame %d: %w", frame, err)
		}
		velocity := make([]float32, len(initial))
		network := func(dst []float32, start, target float32, current []float32) error {
			return g.Flow.Forward(dst, hidden, []float32{start, target}, current)
		}
		if err := LSDDecodeSIMD(normalized, initial, decodeSteps, velocity, network); err != nil {
			return nil, err
		}
		raw := make([]float32, len(normalized))
		if !simd.VecMulTo(raw, normalized, g.Std) || !simd.VecAddTo(raw, raw, g.Mean) {
			return nil, fmt.Errorf("Pocket TTS latent denormalization failed")
		}
		chunk, err := g.Mimi.Decode(raw, 1, mimiState)
		if err != nil {
			return nil, err
		}
		if err := emit(chunk); err != nil {
			return nil, fmt.Errorf("Pocket TTS emit frame %d: %w", frame, err)
		}
		pcm = append(pcm, chunk...)
	}
	if len(pcm) == 0 {
		return nil, fmt.Errorf("Pocket TTS generated no audio")
	}
	return pcm, nil
}
