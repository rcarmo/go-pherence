package diffusiongemma

import "fmt"

// MockDenoiser is a deterministic scaffold denoiser for exercising the
// block-diffusion control flow without model weights. It is not a model.
type MockDenoiser struct {
	VocabSize int
	TokenID   int
	TokenIDs  []int
	Logit     float32
}

func (m MockDenoiser) Denoise(in ForwardInput) (ForwardOutput, error) {
	if m.VocabSize <= 0 {
		return ForwardOutput{}, fmt.Errorf("DiffusionGemma mock denoiser invalid vocab size %d", m.VocabSize)
	}
	tokens := m.TokenIDs
	if len(tokens) == 0 {
		tokens = []int{m.TokenID}
	}
	for _, token := range tokens {
		if token < 0 || token >= m.VocabSize {
			return ForwardOutput{}, fmt.Errorf("DiffusionGemma mock denoiser token %d outside [0,%d)", token, m.VocabSize)
		}
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
		rows[i][tokens[i%len(tokens)]] = logit
	}
	return ForwardOutput{Logits: rows}, nil
}
