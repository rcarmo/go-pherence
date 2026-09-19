package jevlike

import (
	"fmt"
	"github.com/rcarmo/go-pherence/half"
	"math"
)

func roundFeatureRows(rows [][]float32, dtype string) ([][]float32, error) {
	if dtype == "f32" {
		return rows, nil
	}
	for _, row := range rows {
		for i, v := range row {
			row[i] = half.F16ToF32(half.F32ToF16(v))
			if math.IsInf(float64(row[i]), 0) {
				return nil, fmt.Errorf("FP16 feature overflow")
			}
		}
	}
	return rows, nil
}

// FeatureEncoder implements fresh-input execution of exactly the cached feature
// contract without a cache. This is not an identity-by-width fallback.
type FeatureEncoder struct {
	Contract     FeatureContract
	Tokenize     func(string) ([]int, error)
	EncodeTokens func([]int) ([][]float32, error)
}

func (e *FeatureEncoder) FeatureReference() string { return "feature:" + e.Contract.ID() }
func (e *FeatureEncoder) Encode(text string, limit int) ([][]float32, error) {
	if err := e.Contract.Validate(); err != nil {
		return nil, err
	}
	if e.Tokenize == nil || e.EncodeTokens == nil || limit != e.Contract.ContextTokens {
		return nil, fmt.Errorf("invalid live feature encoder")
	}
	rows, err := e.raw(text, limit)
	if err != nil {
		return nil, err
	}
	return roundFeatureRows(rows, e.Contract.DType)
}
func (e *FeatureEncoder) raw(text string, limit int) ([][]float32, error) {
	ids, err := e.Tokenize(text)
	if err != nil {
		return nil, err
	}
	if len(ids) < 1 || len(ids) > limit {
		return nil, fmt.Errorf("live input tokens=%d limit=%d; no truncation", len(ids), limit)
	}
	rows, err := e.EncodeTokens(ids)
	if err != nil {
		return nil, err
	}
	if len(rows) != len(ids) {
		return nil, fmt.Errorf("live token row mismatch")
	}
	return rows, validateFeatureRows(rows, e.Contract.Width)
}
func (e *FeatureEncoder) EncodeOption(text string, limit int) ([]float32, error) {
	if err := e.Contract.Validate(); err != nil {
		return nil, err
	}
	if e.Tokenize == nil || e.EncodeTokens == nil || limit != e.Contract.OptionTokens {
		return nil, fmt.Errorf("invalid live option encoder")
	}
	rows, err := e.raw(text, limit)
	if err != nil {
		return nil, err
	}
	pooled := make([]float32, e.Contract.Width)
	for _, row := range rows {
		for i, v := range row {
			pooled[i] += v / float32(len(rows))
		}
	}
	rounded, err := roundFeatureRows([][]float32{pooled}, e.Contract.DType)
	if err != nil {
		return nil, err
	}
	return rounded[0], nil
}
