package minicpmv

import "fmt"

// VisionEmbeddingCPU composes the executable SigLIP tower, perceiver
// resampler, and existing non-aliasing language-embedding injection boundary.
type VisionEmbeddingCPU struct {
	Vision    *SigLIPVisionCPU
	Resampler *ResamplerCPU
}

func NewVisionEmbeddingCPU(vision *SigLIPVisionCPU, resampler *ResamplerCPU) (*VisionEmbeddingCPU, error) {
	if vision == nil || resampler == nil {
		return nil, fmt.Errorf("MiniCPM-V/O vision embedding runtime requires SigLIP and resampler stages")
	}
	if vision.hidden != resampler.kvDim {
		return nil, fmt.Errorf("MiniCPM-V/O vision/resampler hidden mismatch=%d/%d", vision.hidden, resampler.kvDim)
	}
	return &VisionEmbeddingCPU{Vision: vision, Resampler: resampler}, nil
}

func (m *VisionEmbeddingCPU) EncodeAndInject(pixelValues []float32, shape [4]int, tokenEmbeddings []float32, seqLen, hidden int, plan PromptPlan) ([]float32, EmbeddingInjection, error) {
	meta := EmbeddingInjection{SequenceLength: seqLen, HiddenSize: hidden, Images: len(plan.ImageSpans)}
	if m == nil || m.Vision == nil || m.Resampler == nil {
		return nil, meta, fmt.Errorf("nil MiniCPM-V/O vision embedding runtime")
	}
	if len(plan.ImageSpans) != 1 {
		return nil, meta, fmt.Errorf("MiniCPM-V/O vision embedding runtime expects one image span, got %d", len(plan.ImageSpans))
	}
	vision, err := m.Vision.EncodeImage(pixelValues, shape)
	if err != nil {
		return nil, meta, err
	}
	if len(vision)%m.Vision.hidden != 0 {
		return nil, meta, fmt.Errorf("invalid MiniCPM-V/O SigLIP output values=%d hidden=%d", len(vision), m.Vision.hidden)
	}
	imageEmbeddings, err := m.Resampler.Resample(vision, len(vision)/m.Vision.hidden, m.Vision.hidden)
	if err != nil {
		return nil, meta, err
	}
	return InjectImageEmbeddings(tokenEmbeddings, seqLen, hidden, plan, imageEmbeddings)
}
