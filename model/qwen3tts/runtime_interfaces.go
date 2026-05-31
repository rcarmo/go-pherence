package qwen3tts

import "errors"

var ErrRuntimeNotImplemented = errors.New("qwen3tts runtime execution is not implemented")

// TalkerRuntime is the CPU/reference Talker boundary. Implementations consume a
// validated prompt request and return semantic tokens; the interface exists so
// callers can be wired to checked contracts before generation is implemented.
type TalkerRuntime interface {
	ForwardSemantic(plan RuntimeRequestPlan) ([]uint32, error)
}

// CodePredictorRuntime is the CPU/reference acoustic-code boundary. It consumes
// semantic tokens and returns flattened acoustic codebook IDs for Decoder12Hz.
type CodePredictorRuntime interface {
	PredictAcoustic(plan RuntimeRequestPlan, semantic []uint32) ([]uint32, error)
}

// Decoder12HzRuntime is the CPU/reference waveform boundary. It consumes
// flattened acoustic codebook IDs and returns mono PCM float samples at 24kHz.
type Decoder12HzRuntime interface {
	DecodeWaveform(plan RuntimeRequestPlan, acoustic []uint32) ([]float32, error)
}

// RuntimePipeline groups the three execution stages under one contract. The
// default constructor intentionally returns a not-implemented runtime until the
// CPU/reference paths land.
type RuntimePipeline interface {
	TalkerRuntime
	CodePredictorRuntime
	Decoder12HzRuntime
}

type NotImplementedRuntime struct{}

func NewNotImplementedRuntime() RuntimePipeline { return NotImplementedRuntime{} }

func (NotImplementedRuntime) ForwardSemantic(RuntimeRequestPlan) ([]uint32, error) {
	return nil, ErrRuntimeNotImplemented
}

func (NotImplementedRuntime) PredictAcoustic(RuntimeRequestPlan, []uint32) ([]uint32, error) {
	return nil, ErrRuntimeNotImplemented
}

func (NotImplementedRuntime) DecodeWaveform(RuntimeRequestPlan, []uint32) ([]float32, error) {
	return nil, ErrRuntimeNotImplemented
}
