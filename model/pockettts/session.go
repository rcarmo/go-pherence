package pockettts

import "fmt"

// Session owns mutable FlowLM/Mimi state and request buffers. It is not safe for
// concurrent use; create one session per request/worker.
type Session struct {
	Generator                        *GeneratorCPU
	VoiceTemplate                    *TransformerState
	FlowState                        *TransformerState
	MimiState                        *MimiDecoderState
	MaxFrames                        int
	Normalized, Noise, Velocity, Raw []float32
	FlowRow, Hidden, EOS             []float32
	FlowScratch                      *FlowHeadScratch
}

func NewSession(g *GeneratorCPU, voice *TransformerState, maxFrames int) (*Session, error) {
	if g == nil || voice == nil || maxFrames <= 0 {
		return nil, fmt.Errorf("invalid Pocket TTS session")
	}
	capacity := voice.Position + maxFrames + 64
	if len(voice.Layers) > 0 && voice.Layers[0].Capacity > capacity {
		capacity = voice.Layers[0].Capacity
	}
	flow, err := g.FlowLM.Transformer.NewState(capacity)
	if err != nil {
		return nil, err
	}
	flowScratch, err := g.Flow.NewScratch()
	if err != nil {
		return nil, err
	}
	s := &Session{Generator: g, VoiceTemplate: voice, FlowState: flow, MimiState: g.Mimi.NewStateForFrames(maxFrames), MaxFrames: maxFrames, Normalized: make([]float32, g.FlowLM.Config.Mimi.InnerDim), Noise: make([]float32, g.FlowLM.Config.Mimi.InnerDim), Velocity: make([]float32, g.FlowLM.Config.Mimi.InnerDim), Raw: make([]float32, g.FlowLM.Config.Mimi.InnerDim), FlowRow: make([]float32, g.FlowLM.Config.FlowLM.Transformer.DModel), Hidden: make([]float32, g.FlowLM.Config.FlowLM.Transformer.DModel), EOS: make([]float32, 1), FlowScratch: flowScratch}
	if err := s.Reset(); err != nil {
		return nil, err
	}
	return s, nil
}
func copyTransformerStateInto(dst, src *TransformerState) error {
	if dst == nil || src == nil || len(dst.Layers) != len(src.Layers) {
		return fmt.Errorf("invalid Pocket TTS state reset")
	}
	dst.Position = src.Position
	for i := range src.Layers {
		width := len(dst.Layers[i].Keys) / dst.Layers[i].Capacity
		if src.Layers[i].Length > dst.Layers[i].Capacity || len(src.Layers[i].Keys) < src.Layers[i].Length*width {
			return fmt.Errorf("Pocket TTS state capacity exceeded")
		}
		clear(dst.Layers[i].Keys)
		clear(dst.Layers[i].Values)
		copy(dst.Layers[i].Keys, src.Layers[i].Keys[:src.Layers[i].Length*width])
		copy(dst.Layers[i].Values, src.Layers[i].Values[:src.Layers[i].Length*width])
		dst.Layers[i].Length = src.Layers[i].Length
	}
	return nil
}
func resetMimiState(s *MimiDecoderState) {
	if s == nil {
		return
	}
	clear(s.Upsample.Partial)
	if s.Transformer != nil {
		s.Transformer.Position = 0
		for i := range s.Transformer.Layers {
			clear(s.Transformer.Layers[i].Keys)
			clear(s.Transformer.Layers[i].Values)
			s.Transformer.Layers[i].Length = 0
		}
	}
	clear(s.Initial.Previous)
	s.Initial.First = true
	for i := range s.Stages {
		clear(s.Stages[i].Up.Partial)
		clear(s.Stages[i].Residual.Conv1.Previous)
		s.Stages[i].Residual.Conv1.First = true
		clear(s.Stages[i].Residual.Conv2.Previous)
		s.Stages[i].Residual.Conv2.First = true
	}
	clear(s.Final.Previous)
	s.Final.First = true
}
func (s *Session) Reset() error {
	if s == nil || s.Generator == nil {
		return fmt.Errorf("nil Pocket TTS session")
	}
	if err := copyTransformerStateInto(s.FlowState, s.VoiceTemplate); err != nil {
		return err
	}
	resetMimiState(s.MimiState)
	clear(s.Normalized)
	clear(s.Noise)
	clear(s.Velocity)
	clear(s.Raw)
	clear(s.FlowRow)
	clear(s.Hidden)
	clear(s.EOS)
	return nil
}
