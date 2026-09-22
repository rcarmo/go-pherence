package pockettts

import (
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

type TransformerChunkScratch struct {
	Rows                                                        int
	Norm, QKV, Attention, Projected, Output, FFNorm, Wide, Down []float32
	HiddenA, HiddenB                                            []float32
}

func (m *TransformerCPU) NewChunkScratch(rows int) (*TransformerChunkScratch, error) {
	if m == nil || rows <= 0 || rows > 256 {
		return nil, fmt.Errorf("invalid Pocket TTS transformer chunk rows")
	}
	wide := 0
	for i := range m.Layers {
		wide = max(wide, m.Layers[i].FC1.Out)
	}
	return &TransformerChunkScratch{Rows: rows, Norm: make([]float32, rows*m.Width), QKV: make([]float32, rows*3*m.Width), Attention: make([]float32, rows*m.Width), Projected: make([]float32, rows*m.Width), Output: make([]float32, rows*m.Width), FFNorm: make([]float32, rows*m.Width), Wide: make([]float32, rows*wide), Down: make([]float32, rows*m.Width), HiddenA: make([]float32, rows*m.Width), HiddenB: make([]float32, rows*m.Width)}, nil
}

// ForwardChunkInto advances a fixed row-major chunk through existing KV state.
// It batches linear projections while retaining row-causal attention semantics.
func (m *TransformerCPU) ForwardChunkInto(dst, input []float32, rows int, state *TransformerState, s *TransformerChunkScratch) error {
	if m == nil || state == nil || s == nil || rows <= 0 || rows > s.Rows || len(input) < rows*m.Width || len(dst) < rows*m.Width || state.Position+rows > state.Layers[0].Capacity {
		return fmt.Errorf("invalid Pocket TTS transformer chunk")
	}
	copy(s.HiddenA[:rows*m.Width], input[:rows*m.Width])
	current, next := s.HiddenA[:rows*m.Width], s.HiddenB[:rows*m.Width]
	basePos := state.Position
	for li := range m.Layers {
		l := &m.Layers[li]
		ls := &state.Layers[li]
		if ls.Length != basePos {
			return fmt.Errorf("Pocket TTS chunk state length=%d want=%d", ls.Length, basePos)
		}
		norm := s.Norm[:rows*m.Width]
		if !simd.LayerNormLastAxisTo(norm, current, rows, m.Width, l.Norm1Weight, l.Norm1Bias, 1e-5) {
			return fmt.Errorf("Pocket TTS chunk norm1")
		}
		qkv := s.QKV[:rows*3*m.Width]
		if err := l.InProjection.ForwardRows(qkv, norm, rows); err != nil {
			return err
		}
		for row := 0; row < rows; row++ {
			qrow := qkv[row*3*m.Width : row*3*m.Width+m.Width]
			krow := qkv[row*3*m.Width+m.Width : row*3*m.Width+2*m.Width]
			vrow := qkv[row*3*m.Width+2*m.Width : (row+1)*3*m.Width]
			pos := basePos + row
			key := ls.Keys[pos*m.Width : (pos+1)*m.Width]
			value := ls.Values[pos*m.Width : (pos+1)*m.Width]
			copy(key, krow)
			copy(value, vrow)
			if !applyPocketRoPEScratch(qrow, state.Rope, pos, m.Heads, m.HeadDim, state) || !applyPocketRoPEScratch(key, state.Rope, pos, m.Heads, m.HeadDim, state) {
				return fmt.Errorf("Pocket TTS chunk RoPE")
			}
			ls.Length++
			attn := s.Attention[row*m.Width : (row+1)*m.Width]
			start := 0
			if m.Context > 0 && ls.Length > m.Context {
				start = ls.Length - m.Context
			}
			scores := ls.Scores[:ls.Length-start]
			if !simd.GQAAttentionScaleTo(attn, scores, qrow, ls.Keys[start*m.Width:ls.Length*m.Width], ls.Values[start*m.Width:ls.Length*m.Width], ls.Length-start, m.Heads, m.Heads, m.HeadDim, float32(1/math.Sqrt(float64(m.HeadDim)))) {
				return fmt.Errorf("Pocket TTS chunk attention")
			}
		}
		projected := s.Projected[:rows*m.Width]
		if err := l.OutProjection.ForwardRows(projected, s.Attention[:rows*m.Width], rows); err != nil {
			return err
		}
		for row := 0; row < rows; row++ {
			p := projected[row*m.Width : (row+1)*m.Width]
			if l.LayerScale1 != nil && !simd.VecMulTo(p, p, l.LayerScale1) {
				return fmt.Errorf("Pocket TTS chunk layer scale1")
			}
			if !simd.VecAddTo(s.Output[row*m.Width:(row+1)*m.Width], current[row*m.Width:(row+1)*m.Width], p) {
				return fmt.Errorf("Pocket TTS chunk residual1")
			}
		}
		if !simd.LayerNormLastAxisTo(s.FFNorm[:rows*m.Width], s.Output[:rows*m.Width], rows, m.Width, l.Norm2Weight, l.Norm2Bias, 1e-5) {
			return fmt.Errorf("Pocket TTS chunk norm2")
		}
		wide := s.Wide[:rows*l.FC1.Out]
		if err := l.FC1.ForwardRows(wide, s.FFNorm[:rows*m.Width], rows); err != nil {
			return err
		}
		if !simd.GELUTanhTo(wide, wide) {
			return fmt.Errorf("Pocket TTS chunk GELU")
		}
		down := s.Down[:rows*m.Width]
		if err := l.FC2.ForwardRows(down, wide, rows); err != nil {
			return err
		}
		for row := 0; row < rows; row++ {
			d := down[row*m.Width : (row+1)*m.Width]
			if l.LayerScale2 != nil && !simd.VecMulTo(d, d, l.LayerScale2) {
				return fmt.Errorf("Pocket TTS chunk layer scale2")
			}
			if !simd.VecAddTo(next[row*m.Width:(row+1)*m.Width], s.Output[row*m.Width:(row+1)*m.Width], d) {
				return fmt.Errorf("Pocket TTS chunk residual2")
			}
		}
		current, next = next, current
	}
	state.Position += rows
	if m.FinalWeight != nil {
		if !simd.LayerNormLastAxisTo(dst, current, rows, m.Width, m.FinalWeight, m.FinalBias, 1e-5) {
			return fmt.Errorf("Pocket TTS chunk final norm")
		}
	} else {
		copy(dst[:rows*m.Width], current)
	}
	return nil
}
