package pockettts

import (
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

type TransformerLayerState struct {
	Keys, Values                          []float32
	Norm, QKV, Attention, Scores          []float32
	Projected, Output, FFNorm, Wide, Down []float32
	Length, Capacity                      int
}

type TransformerState struct {
	Layers                               []TransformerLayerState
	Rope                                 []float32
	RopeCos, RopeSin, RopeReal, RopeImag []float32
	RopeRC, RopeIS, RopeIC, RopeRS       []float32
	HiddenA, HiddenB                     []float32
	Position                             int
}

func (m *TransformerCPU) NewState(capacity int) (*TransformerState, error) {
	if m == nil || capacity <= 0 || m.Width <= 0 || len(m.Layers) == 0 || capacity > int(^uint(0)>>1)/m.Width {
		return nil, fmt.Errorf("invalid Pocket TTS transformer state capacity %d", capacity)
	}
	half := m.HeadDim / 2
	state := &TransformerState{Layers: make([]TransformerLayerState, len(m.Layers)), Rope: simd.BuildRoPEFreqs(capacity, half, m.HeadDim, m.MaxPeriod), RopeCos: make([]float32, half), RopeSin: make([]float32, half), RopeReal: make([]float32, half), RopeImag: make([]float32, half), RopeRC: make([]float32, half), RopeIS: make([]float32, half), RopeIC: make([]float32, half), RopeRS: make([]float32, half), HiddenA: make([]float32, m.Width), HiddenB: make([]float32, m.Width)}
	for i := range state.Layers {
		wide := m.Layers[i].FC1.Out
		state.Layers[i] = TransformerLayerState{Keys: make([]float32, capacity*m.Width), Values: make([]float32, capacity*m.Width), Norm: make([]float32, m.Width), QKV: make([]float32, 3*m.Width), Attention: make([]float32, m.Width), Scores: make([]float32, capacity), Projected: make([]float32, m.Width), Output: make([]float32, m.Width), FFNorm: make([]float32, m.Width), Wide: make([]float32, wide), Down: make([]float32, m.Width), Capacity: capacity}
	}
	return state, nil
}

// Step is the allocating convenience API. Use StepInto on warm paths.
func (m *TransformerCPU) Step(input []float32, state *TransformerState) ([]float32, error) {
	out := make([]float32, m.Width)
	if err := m.StepInto(out, input, state); err != nil {
		return nil, err
	}
	return out, nil
}

// StepInto advances one row through request-owned causal KV and scratch state.
func (m *TransformerCPU) StepInto(dst, input []float32, state *TransformerState) error {
	if m == nil || state == nil || len(dst) != m.Width || len(input) != m.Width || len(state.Layers) != len(m.Layers) || state.Position < 0 || state.Position >= state.Layers[0].Capacity || len(state.Rope) < (state.Position+1)*m.HeadDim {
		return fmt.Errorf("invalid Pocket TTS transformer step")
	}
	copy(state.HiddenA, input)
	current, next := state.HiddenA, state.HiddenB
	for i := range m.Layers {
		if err := m.Layers[i].stepInto(next, current, state.Position, m.Width, m.Heads, m.HeadDim, m.Context, state.Rope, &state.Layers[i], state); err != nil {
			return fmt.Errorf("Pocket TTS transformer step layer %d: %w", i, err)
		}
		current, next = next, current
	}
	state.Position++
	if m.FinalWeight != nil {
		if !simd.LayerNormLastAxisTo(dst, current, 1, m.Width, m.FinalWeight, m.FinalBias, 1e-5) {
			return fmt.Errorf("Pocket TTS streaming final LayerNorm failed")
		}
	} else {
		copy(dst, current)
	}
	return nil
}

func (l TransformerLayerCPU) stepInto(dst, input []float32, position, width, heads, headDim, context int, rope []float32, state *TransformerLayerState, transformer *TransformerState) error {
	if state == nil || state.Length != position || position >= state.Capacity || len(state.Keys) < state.Capacity*width || len(state.Values) < state.Capacity*width || len(dst) != width {
		return fmt.Errorf("invalid Pocket TTS layer state position=%d length=%d capacity=%d", position, state.Length, state.Capacity)
	}
	if !simd.LayerNormLastAxisTo(state.Norm, input, 1, width, l.Norm1Weight, l.Norm1Bias, 1e-5) {
		return fmt.Errorf("attention LayerNorm failed")
	}
	if err := l.InProjection.Forward(state.QKV, state.Norm); err != nil {
		return err
	}
	q := state.QKV[:width]
	key := state.Keys[position*width : (position+1)*width]
	value := state.Values[position*width : (position+1)*width]
	copy(key, state.QKV[width:2*width])
	copy(value, state.QKV[2*width:])
	if !applyPocketRoPEScratch(q, rope, position, heads, headDim, transformer) || !applyPocketRoPEScratch(key, rope, position, heads, headDim, transformer) {
		return fmt.Errorf("RoPE failed")
	}
	state.Length++
	start := 0
	if context > 0 && state.Length > context {
		start = state.Length - context
	}
	scores := state.Scores[:state.Length-start]
	if !simd.GQAAttentionScaleTo(state.Attention, scores, q, state.Keys[start*width:state.Length*width], state.Values[start*width:state.Length*width], state.Length-start, heads, heads, headDim, float32(1/math.Sqrt(float64(headDim)))) {
		return fmt.Errorf("streaming attention failed")
	}
	if err := l.OutProjection.Forward(state.Projected, state.Attention); err != nil {
		return err
	}
	if l.LayerScale1 != nil && !simd.VecMulTo(state.Projected, state.Projected, l.LayerScale1) {
		return fmt.Errorf("attention layer scale failed")
	}
	if !simd.VecAddTo(state.Output, input, state.Projected) {
		return fmt.Errorf("attention residual failed")
	}
	if !simd.LayerNormLastAxisTo(state.FFNorm, state.Output, 1, width, l.Norm2Weight, l.Norm2Bias, 1e-5) {
		return fmt.Errorf("FFN LayerNorm failed")
	}
	if err := l.FC1.Forward(state.Wide, state.FFNorm); err != nil {
		return err
	}
	if !simd.GELUTanhTo(state.Wide, state.Wide) {
		return fmt.Errorf("tanh GELU failed")
	}
	if err := l.FC2.Forward(state.Down, state.Wide); err != nil {
		return err
	}
	if l.LayerScale2 != nil && !simd.VecMulTo(state.Down, state.Down, l.LayerScale2) {
		return fmt.Errorf("FFN layer scale failed")
	}
	if !simd.VecAddTo(dst, state.Output, state.Down) {
		return fmt.Errorf("FFN residual failed")
	}
	return nil
}

func applyPocketRoPEScratch(x, freqs []float32, pos, heads, headDim int, s *TransformerState) bool {
	pairs := headDim / 2
	if s == nil || pos < 0 || heads <= 0 || headDim <= 0 || headDim%2 != 0 || len(x) < heads*headDim || len(freqs) < (pos+1)*pairs*2 || len(s.RopeCos) < pairs {
		return false
	}
	for i := 0; i < pairs; i++ {
		off := (pos*pairs + i) * 2
		s.RopeCos[i], s.RopeSin[i] = freqs[off], freqs[off+1]
	}
	for head := 0; head < heads; head++ {
		base := head * headDim
		for i := 0; i < pairs; i++ {
			s.RopeReal[i], s.RopeImag[i] = x[base+2*i], x[base+2*i+1]
		}
		if !simd.VecMulTo(s.RopeRC, s.RopeReal, s.RopeCos) || !simd.VecMulTo(s.RopeIS, s.RopeImag, s.RopeSin) || !simd.VecMulTo(s.RopeIC, s.RopeImag, s.RopeCos) || !simd.VecMulTo(s.RopeRS, s.RopeReal, s.RopeSin) || !simd.VecScaleAddTo(s.RopeRC, s.RopeRC, s.RopeIS, -1) || !simd.VecAddTo(s.RopeIC, s.RopeIC, s.RopeRS) {
			return false
		}
		for i := 0; i < pairs; i++ {
			x[base+2*i], x[base+2*i+1] = s.RopeRC[i], s.RopeIC[i]
		}
	}
	return true
}
