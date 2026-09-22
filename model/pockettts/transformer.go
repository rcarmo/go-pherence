package pockettts

import (
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

type TransformerLayerCPU struct {
	Norm1Weight, Norm1Bias      []float32
	Norm2Weight, Norm2Bias      []float32
	InProjection, OutProjection LinearF32
	FC1, FC2                    LinearF32
	LayerScale1, LayerScale2    []float32
}

type TransformerCPU struct {
	Width, Heads, HeadDim  int
	Context                int
	MaxPeriod              float64
	Layers                 []TransformerLayerCPU
	FinalWeight, FinalBias []float32
}

func (m *TransformerCPU) Forward(sequence []float32, rows int) ([]float32, error) {
	if m == nil || rows <= 0 || m.Width <= 0 || m.Heads <= 0 || m.HeadDim <= 0 || m.Width != m.Heads*m.HeadDim || len(sequence) != rows*m.Width || len(m.Layers) == 0 {
		return nil, fmt.Errorf("invalid Pocket TTS transformer input")
	}
	hidden := append([]float32(nil), sequence...)
	rope := simd.BuildRoPEFreqs(rows, m.HeadDim/2, m.HeadDim, m.MaxPeriod)
	for i := range m.Layers {
		var err error
		hidden, err = m.Layers[i].forward(hidden, rows, m.Width, m.Heads, m.HeadDim, rope)
		if err != nil {
			return nil, fmt.Errorf("Pocket TTS transformer layer %d: %w", i, err)
		}
	}
	if m.FinalWeight != nil {
		out := make([]float32, len(hidden))
		if !simd.LayerNormLastAxisTo(out, hidden, rows, m.Width, m.FinalWeight, m.FinalBias, 1e-5) {
			return nil, fmt.Errorf("Pocket TTS final LayerNorm failed")
		}
		hidden = out
	}
	return hidden, nil
}

func (l TransformerLayerCPU) forward(input []float32, rows, width, heads, headDim int, rope []float32) ([]float32, error) {
	norm := make([]float32, len(input))
	if !simd.LayerNormLastAxisTo(norm, input, rows, width, l.Norm1Weight, l.Norm1Bias, 1e-5) {
		return nil, fmt.Errorf("attention LayerNorm failed")
	}
	qkv := make([]float32, 3*width)
	q, k, v := make([]float32, len(input)), make([]float32, len(input)), make([]float32, len(input))
	for row := 0; row < rows; row++ {
		if err := l.InProjection.Forward(qkv, norm[row*width:(row+1)*width]); err != nil {
			return nil, err
		}
		copy(q[row*width:(row+1)*width], qkv[:width])
		copy(k[row*width:(row+1)*width], qkv[width:2*width])
		copy(v[row*width:(row+1)*width], qkv[2*width:])
		if !applyPocketRoPETo(q[row*width:(row+1)*width], rope, row, heads, headDim) || !applyPocketRoPETo(k[row*width:(row+1)*width], rope, row, heads, headDim) {
			return nil, fmt.Errorf("RoPE failed")
		}
	}
	attention := make([]float32, len(input))
	scores := make([]float32, rows)
	scale := float32(1 / math.Sqrt(float64(headDim)))
	for row := 0; row < rows; row++ {
		for head := 0; head < heads; head++ {
			qh := q[row*width+head*headDim : row*width+(head+1)*headDim]
			for key := 0; key <= row; key++ {
				scores[key] = simd.Sdot(qh, k[key*width+head*headDim:key*width+(head+1)*headDim]) * scale
			}
			if !simd.SoftmaxInPlace(scores[:row+1]) {
				return nil, fmt.Errorf("softmax failed")
			}
			dst := attention[row*width+head*headDim : row*width+(head+1)*headDim]
			for key := 0; key <= row; key++ {
				simd.VecScaleAdd(dst, dst, v[key*width+head*headDim:key*width+(head+1)*headDim], scores[key])
			}
		}
	}
	out := make([]float32, len(input))
	projected := make([]float32, width)
	for row := 0; row < rows; row++ {
		if err := l.OutProjection.Forward(projected, attention[row*width:(row+1)*width]); err != nil {
			return nil, err
		}
		update := projected
		if l.LayerScale1 != nil {
			update = append([]float32(nil), projected...)
			if !simd.VecMulTo(update, update, l.LayerScale1) {
				return nil, fmt.Errorf("attention layer scale failed")
			}
		}
		if !simd.VecAddTo(out[row*width:(row+1)*width], input[row*width:(row+1)*width], update) {
			return nil, fmt.Errorf("attention residual failed")
		}
	}
	ffnorm := make([]float32, len(out))
	if !simd.LayerNormLastAxisTo(ffnorm, out, rows, width, l.Norm2Weight, l.Norm2Bias, 1e-5) {
		return nil, fmt.Errorf("FFN LayerNorm failed")
	}
	wide := make([]float32, l.FC1.Out)
	down := make([]float32, width)
	for row := 0; row < rows; row++ {
		if err := l.FC1.Forward(wide, ffnorm[row*width:(row+1)*width]); err != nil {
			return nil, err
		}
		if !simd.GELUTanhTo(wide, wide) {
			return nil, fmt.Errorf("tanh GELU failed")
		}
		if err := l.FC2.Forward(down, wide); err != nil {
			return nil, err
		}
		update := down
		if l.LayerScale2 != nil {
			update = append([]float32(nil), down...)
			if !simd.VecMulTo(update, update, l.LayerScale2) {
				return nil, fmt.Errorf("FFN layer scale failed")
			}
		}
		if !simd.VecAddTo(out[row*width:(row+1)*width], out[row*width:(row+1)*width], update) {
			return nil, fmt.Errorf("FFN residual failed")
		}
	}
	return out, nil
}

// applyPocketRoPETo rotates adjacent complex pairs, matching upstream
// view(..., head_dim/2, 2). Shared LLM RoPE uses a split-half layout instead.
// Pair arithmetic dispatches through SIMD vector operations; gather/scatter
// only changes the layout around those operations.
func applyPocketRoPETo(x, freqs []float32, pos, heads, headDim int) bool {
	pairs := headDim / 2
	if pos < 0 || heads <= 0 || headDim <= 0 || headDim%2 != 0 || len(x) < heads*headDim || len(freqs) < (pos+1)*pairs*2 {
		return false
	}
	cosines, sines := make([]float32, pairs), make([]float32, pairs)
	for i := 0; i < pairs; i++ {
		off := (pos*pairs + i) * 2
		cosines[i], sines[i] = freqs[off], freqs[off+1]
	}
	real, imag := make([]float32, pairs), make([]float32, pairs)
	rc, is, ic, rs := make([]float32, pairs), make([]float32, pairs), make([]float32, pairs), make([]float32, pairs)
	for head := 0; head < heads; head++ {
		base := head * headDim
		for i := 0; i < pairs; i++ {
			real[i], imag[i] = x[base+2*i], x[base+2*i+1]
		}
		if !simd.VecMulTo(rc, real, cosines) || !simd.VecMulTo(is, imag, sines) || !simd.VecMulTo(ic, imag, cosines) || !simd.VecMulTo(rs, real, sines) || !simd.VecScaleAddTo(rc, rc, is, -1) || !simd.VecAddTo(ic, ic, rs) {
			return false
		}
		for i := 0; i < pairs; i++ {
			x[base+2*i], x[base+2*i+1] = rc[i], ic[i]
		}
	}
	return true
}
