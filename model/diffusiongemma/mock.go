package diffusiongemma

import "fmt"

// MockDenoiser is a deterministic scaffold denoiser for exercising the
// block-diffusion control flow without model weights. It is not a model.
type MockDenoiser struct {
	VocabSize int
	TokenID   int
	Logit     float32
}

func (m MockDenoiser) Denoise(in ForwardInput) (ForwardOutput, error) {
	if m.VocabSize <= 0 {
		return ForwardOutput{}, fmt.Errorf("DiffusionGemma mock denoiser invalid vocab size %d", m.VocabSize)
	}
	if m.TokenID < 0 || m.TokenID >= m.VocabSize {
		return ForwardOutput{}, fmt.Errorf("DiffusionGemma mock denoiser token %d outside [0,%d)", m.TokenID, m.VocabSize)
	}
	if len(in.Canvas) == 0 {
		return ForwardOutput{}, fmt.Errorf("DiffusionGemma mock denoiser empty canvas")
	}
	logit := m.Logit
	if logit == 0 {
		logit = 10
	}
	rows := make([][]float32, len(in.Canvas))
	for i := range rows {
		rows[i] = make([]float32, m.VocabSize)
		rows[i][m.TokenID] = logit
	}
	return ForwardOutput{Logits: rows}, nil
}
