package gliner2

import (
	"fmt"
	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"math"
)

// MaskLogit is upstream's finite sentinel, safe when proposal terms are added.
const MaskLogit float32 = -1e4

// BoundaryQueryHead implements inference for upstream boundary/heads.py.
type BoundaryQueryHead struct {
	StartBoundary, StartQuery Linear
	EndBoundary, EndQuery     Linear
	InsideText, InsideQuery   Linear
}

type BoundaryMarginals struct {
	StartLogits, EndLogits, InsideLogits [][]float32
	InsidePrefix                         [][]float32
	InsidePrefixMean                     []float32
}

func (h BoundaryQueryHead) Validate() error {
	projections := []Linear{h.StartBoundary, h.StartQuery, h.EndBoundary, h.EndQuery, h.InsideText, h.InsideQuery}
	for _, p := range projections {
		if err := p.Validate(); err != nil {
			return err
		}
		if p.OutDim != h.StartBoundary.OutDim {
			return fmt.Errorf("boundary projection output dimensions differ")
		}
	}
	if h.StartBoundary.InDim != h.EndBoundary.InDim || h.StartBoundary.InDim != h.StartBoundary.OutDim {
		return fmt.Errorf("invalid boundary state width")
	}
	if h.StartQuery.InDim != h.EndQuery.InDim || h.StartQuery.InDim != h.InsideQuery.InDim {
		return fmt.Errorf("query dimensions differ")
	}
	return nil
}

// Forward scores a single padded document and its padded schema queries.
func (h BoundaryQueryHead) Forward(boundary, text, queries [][]float32, boundaryMask, textMask, queryMask []bool) (BoundaryMarginals, error) {
	if err := h.Validate(); err != nil {
		return BoundaryMarginals{}, err
	}
	if len(boundary) != len(text)+1 || len(boundaryMask) != len(boundary) || len(textMask) != len(text) || len(queryMask) != len(queries) {
		return BoundaryMarginals{}, fmt.Errorf("boundary query mask/state shape mismatch")
	}
	project := func(p Linear, rows [][]float32) ([][]float32, error) {
		flat, err := flattenRows(rows, p.InDim, "query head")
		if err != nil {
			return nil, err
		}
		out := make([]float32, len(rows)*p.OutDim)
		if err = p.ApplyBatch(flat, out, len(rows)); err != nil {
			return nil, err
		}
		return rowsFromFlat(out, len(rows), p.OutDim), nil
	}
	sb, err := project(h.StartBoundary, boundary)
	if err != nil {
		return BoundaryMarginals{}, err
	}
	eb, err := project(h.EndBoundary, boundary)
	if err != nil {
		return BoundaryMarginals{}, err
	}
	it, err := project(h.InsideText, text)
	if err != nil {
		return BoundaryMarginals{}, err
	}
	sq, err := project(h.StartQuery, queries)
	if err != nil {
		return BoundaryMarginals{}, err
	}
	eq, err := project(h.EndQuery, queries)
	if err != nil {
		return BoundaryMarginals{}, err
	}
	iq, err := project(h.InsideQuery, queries)
	if err != nil {
		return BoundaryMarginals{}, err
	}
	result := BoundaryMarginals{StartLogits: make([][]float32, len(queries)), EndLogits: make([][]float32, len(queries)), InsideLogits: make([][]float32, len(queries)), InsidePrefix: make([][]float32, len(queries)), InsidePrefixMean: make([]float32, len(queries))}
	scale := float32(1 / math.Sqrt(float64(h.StartBoundary.OutDim)))
	for q := range queries {
		result.StartLogits[q] = make([]float32, len(boundary))
		result.EndLogits[q] = make([]float32, len(boundary))
		for i := range boundary {
			s, e := MaskLogit, MaskLogit
			if queryMask[q] && boundaryMask[i] {
				s = simd.Sdot(sb[i], sq[q]) * scale
				e = simd.Sdot(eb[i], eq[q]) * scale
			}
			result.StartLogits[q][i] = s
			result.EndLogits[q][i] = e
		}
		result.InsideLogits[q] = make([]float32, len(text))
		result.InsidePrefix[q] = make([]float32, len(text)+1)
		var sum float32
		count := 0
		for i := range text {
			v := MaskLogit
			if queryMask[q] && textMask[i] {
				v = simd.Sdot(it[i], iq[q]) * scale
				sum += v
				count++
			}
			result.InsideLogits[q][i] = v
		}
		mean := sum / float32(max(count, 1))
		result.InsidePrefixMean[q] = mean
		for i := range text {
			v := result.InsidePrefix[q][i]
			if queryMask[q] && textMask[i] {
				v += result.InsideLogits[q][i] - mean
			}
			result.InsidePrefix[q][i+1] = v
		}
	}
	return result, nil
}
