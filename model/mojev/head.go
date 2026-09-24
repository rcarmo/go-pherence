package mojev

import (
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

// HeadWeights holds owned F32 affine and bias-free projection weights.
// The encoder remains a separate dependency; this head consumes its pre-norm rows.
type HeadWeights struct {
	Width, Rank                         int
	NormWeight, NormBias                []float32
	ContextProjection, OptionProjection []float32 // row-major [rank,width]
}

// NewHeadWeights copies and checks the scorer head before it is used.
func NewHeadWeights(width, rank int, gamma, beta, context, option []float32) (*HeadWeights, error) {
	if width <= 0 || width > 4096 || rank <= 0 || rank > 4096 ||
		len(gamma) != width || len(beta) != width ||
		len(context) != rank*width || len(option) != rank*width {
		return nil, fmt.Errorf("mojev: invalid head dimensions")
	}
	for _, slice := range [][]float32{gamma, beta, context, option} {
		for _, v := range slice {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return nil, fmt.Errorf("mojev: non-finite head weight")
			}
		}
	}
	return &HeadWeights{width, rank, append([]float32(nil), gamma...), append([]float32(nil), beta...),
		append([]float32(nil), context...), append([]float32(nil), option...)}, nil
}

// ScoreHidden computes the F32 released-head readout over pre-LayerNorm encoder
// rows and span masks. Absent options receive math.MaxFloat32's negative value.
// It never mutates inputs and returns no partial logits on an error.
func (h *HeadWeights) ScoreHidden(hidden []float32, state []bool, questions [][]bool, candidates [][][]bool, optionMask [][]bool) ([][]float32, error) {
	if h == nil || h.Width <= 0 || h.Width > 4096 || h.Rank <= 0 || h.Rank > 4096 ||
		len(h.NormWeight) != h.Width || len(h.NormBias) != h.Width ||
		len(h.ContextProjection) != h.Width*h.Rank || len(h.OptionProjection) != h.Width*h.Rank {
		return nil, fmt.Errorf("mojev: uninitialised head")
	}
	owners, err := treeOwners(state, questions, candidates)
	if err != nil {
		return nil, err
	}
	length := len(owners)
	if len(hidden) != length*h.Width || len(optionMask) != len(questions) {
		return nil, fmt.Errorf("mojev: invalid head input geometry")
	}
	for f := range questions {
		if len(optionMask[f]) != len(candidates[f]) {
			return nil, fmt.Errorf("mojev: invalid option mask")
		}
	}
	for _, slice := range [][]float32{h.NormWeight, h.NormBias, h.ContextProjection, h.OptionProjection, hidden} {
		for _, v := range slice {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return nil, fmt.Errorf("mojev: non-finite head input")
			}
		}
	}
	// Keep all intermediate F32 rows request-local. The SIMD operation has a
	// checked Go fallback, while the readout stays independent of the encoder.
	// Pool one span at a time. Each span has a unique owner, so the same
	// hidden row never needs LayerNorm twice. Padding contributes nothing.
	row := make([]float32, h.Width)
	pool := func(dst []float32, span []bool) error {
		clear(dst)
		count := float32(0)
		for pos, active := range span {
			if !active {
				continue
			}
			if !simd.LayerNormLastAxisTo(row, hidden[pos*h.Width:(pos+1)*h.Width], 1, h.Width, h.NormWeight, h.NormBias, 1e-5) {
				return fmt.Errorf("mojev: LayerNorm rejected validated input")
			}
			count++
			for j, v := range row {
				dst[j] += v
			}
		}
		if count > 0 {
			for j := range dst {
				dst[j] /= count
			}
		}
		return nil
	}
	mean := make([]float32, h.Width)
	context, question, key := make([]float32, h.Rank), make([]float32, h.Rank), make([]float32, h.Rank)
	if err := pool(mean, state); err != nil {
		return nil, err
	}
	simd.GemvRows(context, mean, h.ContextProjection, h.Rank, h.Width)
	logits := make([][]float32, len(questions))
	for f, q := range questions {
		if err := pool(mean, q); err != nil {
			return nil, err
		}
		simd.GemvRows(question, mean, h.ContextProjection, h.Rank, h.Width)
		logits[f] = make([]float32, len(candidates[f]))
		for n, c := range candidates[f] {
			if !optionMask[f][n] {
				logits[f][n] = -math.MaxFloat32
				continue
			}
			if err := pool(mean, c); err != nil {
				return nil, err
			}
			simd.GemvRows(key, mean, h.OptionProjection, h.Rank, h.Width)
			var dot float32
			for r, v := range key {
				dot += (context[r] + question[r]) * v
			}
			logits[f][n] = dot / float32(math.Sqrt(float64(h.Rank)))
		}
	}
	return logits, nil
}
