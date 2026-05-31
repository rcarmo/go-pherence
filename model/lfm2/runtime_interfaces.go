package lfm2

import "errors"

var ErrRuntimeNotImplemented = errors.New("lfm2 runtime execution is not implemented")

// EmbeddingRuntime is the CPU/reference token embedding boundary for LFM2.
type EmbeddingRuntime interface {
	Embed(plan RuntimeRequestPlan, tokens []uint32) ([]float32, error)
}

// ConvRuntime is the CPU/reference short-convolution state boundary. It updates
// conv_L_cache state for conv layers and returns hidden activations.
type ConvRuntime interface {
	ForwardConv(plan RuntimeRequestPlan, hidden []float32) ([]float32, error)
}

// AttentionRuntime is the CPU/reference full-attention boundary for LFM2.
type AttentionRuntime interface {
	ForwardAttention(plan RuntimeRequestPlan, hidden []float32) ([]float32, error)
}

// MoERuntime is the CPU/reference router/top-k/expert FFN boundary.
type MoERuntime interface {
	ForwardMoE(plan RuntimeRequestPlan, hidden []float32) ([]float32, error)
}

// GenerationRuntime is the future end-to-end LFM2 generation boundary.
type GenerationRuntime interface {
	EmbeddingRuntime
	ConvRuntime
	AttentionRuntime
	MoERuntime
	Generate(plan RuntimeRequestPlan) ([]uint32, error)
}

type NotImplementedRuntime struct{}

func NewNotImplementedRuntime() GenerationRuntime { return NotImplementedRuntime{} }

func (NotImplementedRuntime) Embed(RuntimeRequestPlan, []uint32) ([]float32, error) {
	return nil, ErrRuntimeNotImplemented
}

func (NotImplementedRuntime) ForwardConv(RuntimeRequestPlan, []float32) ([]float32, error) {
	return nil, ErrRuntimeNotImplemented
}

func (NotImplementedRuntime) ForwardAttention(RuntimeRequestPlan, []float32) ([]float32, error) {
	return nil, ErrRuntimeNotImplemented
}

func (NotImplementedRuntime) ForwardMoE(RuntimeRequestPlan, []float32) ([]float32, error) {
	return nil, ErrRuntimeNotImplemented
}

func (NotImplementedRuntime) Generate(RuntimeRequestPlan) ([]uint32, error) {
	return nil, ErrRuntimeNotImplemented
}
