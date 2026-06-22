package minicpmv

import "errors"

var ErrRuntimeNotImplemented = errors.New("MiniCPM-V/O runtime tensor execution is not implemented")

type VisionTower interface {
	EncodeImage(pixelValues []float32, shape [4]int) ([]float32, error)
}

type Resampler interface {
	Resample(visionTokens []float32, visionTokensCount, visionHidden int) ([]float32, error)
}

type TextBackbone interface {
	GenerateFromEmbeddings(embeddings []float32, seqLen, hidden, maxNewTokens int) ([]int, error)
}

type AudioEncoder interface {
	EncodeAudio(features []float32, frames, featureSize int) ([]float32, error)
}

type PendingRuntime struct{}

func (PendingRuntime) EncodeImage(_ []float32, _ [4]int) ([]float32, error) {
	return nil, ErrRuntimeNotImplemented
}

func (PendingRuntime) Resample(_ []float32, _, _ int) ([]float32, error) {
	return nil, ErrRuntimeNotImplemented
}

func (PendingRuntime) GenerateFromEmbeddings(_ []float32, _, _, _ int) ([]int, error) {
	return nil, ErrRuntimeNotImplemented
}

func (PendingRuntime) EncodeAudio(_ []float32, _, _ int) ([]float32, error) {
	return nil, ErrRuntimeNotImplemented
}

type RuntimeInterfaces struct {
	Vision    VisionTower  `json:"-"`
	Resampler Resampler    `json:"-"`
	Text      TextBackbone `json:"-"`
	Audio     AudioEncoder `json:"-"`
}

func NewPendingRuntimeInterfaces() RuntimeInterfaces {
	pending := PendingRuntime{}
	return RuntimeInterfaces{Vision: pending, Resampler: pending, Text: pending, Audio: pending}
}
