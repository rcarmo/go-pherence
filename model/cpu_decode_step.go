package model

import (
	"fmt"

	"github.com/rcarmo/go-pherence/backends/simd/runtime"
)

// FinishCPUActivation applies the model's final decode norm and returns a copy
// of the final activation without computing LM-head logits. It mutates hidden
// in the same way FinishCPUDecodeStep does before logits.
func (m *LlamaModel) FinishCPUActivation(hidden []float32) ([]float32, error) {
	if m == nil {
		return nil, fmt.Errorf("nil model")
	}
	cfg := m.Config
	if cfg.HiddenSize <= 0 {
		return nil, fmt.Errorf("invalid decode hidden=%d", cfg.HiddenSize)
	}
	if len(hidden) != cfg.HiddenSize {
		return nil, fmt.Errorf("hidden len=%d, want %d", len(hidden), cfg.HiddenSize)
	}
	if m.Norm == nil {
		return nil, fmt.Errorf("model final norm is not loaded")
	}
	norm := m.Norm.Data()
	if len(norm) < cfg.HiddenSize {
		return nil, fmt.Errorf("final norm len=%d, want at least %d", len(norm), cfg.HiddenSize)
	}
	if cfg.ModelType == "gemma3_text" {
		simd.RMSNormBF16(hidden, norm, float32(cfg.RMSNormEps))
	} else {
		rmsNormInPlace(hidden, norm, float32(cfg.RMSNormEps))
	}
	return append([]float32(nil), hidden...), nil
}

// FinishCPUDecodeStep applies the model's final decode norm, computes LM-head
// logits, and returns the greedy token. It mutates hidden in the same way the
// historical Generate path did, and returns a copy of that final activation so
// callers can retain it independently of scratch buffers.
func (m *LlamaModel) FinishCPUDecodeStep(hidden []float32) (finalActivation []float32, logits []float32, token int, err error) {
	if m == nil {
		return nil, nil, 0, fmt.Errorf("nil model")
	}
	cfg := m.Config
	if cfg.HiddenSize <= 0 || cfg.VocabSize <= 0 {
		return nil, nil, 0, fmt.Errorf("invalid decode dims hidden=%d vocab=%d", cfg.HiddenSize, cfg.VocabSize)
	}
	if len(hidden) != cfg.HiddenSize {
		return nil, nil, 0, fmt.Errorf("hidden len=%d, want %d", len(hidden), cfg.HiddenSize)
	}
	finalActivation, err = m.FinishCPUActivation(hidden)
	if err != nil {
		return nil, nil, 0, err
	}
	logits = make([]float32, cfg.VocabSize)
	if err := m.LMHeadLogitsInto(logits, hidden); err != nil {
		return nil, nil, 0, err
	}
	token, _, err = ArgmaxLogits(logits)
	if err != nil {
		return nil, nil, 0, err
	}
	return finalActivation, logits, token, nil
}

// FinishCPUDecodeBatch applies the final norm to a batch of hidden rows and
// computes all LM-head logits with one checked SIMD SGEMM when possible. It is
// the verifier-tail lowering used by Gemma4 MTP batch verification before the
// full layer stack is vectorized.
func (m *LlamaModel) FinishCPUDecodeBatch(hiddenRows [][]float32) (finalActivations [][]float32, logitsRows [][]float32, tokens []int, err error) {
	if m == nil {
		return nil, nil, nil, fmt.Errorf("nil model")
	}
	cfg := m.Config
	B := len(hiddenRows)
	if B <= 0 {
		return nil, nil, nil, fmt.Errorf("empty decode batch")
	}
	if cfg.HiddenSize <= 0 || cfg.VocabSize <= 0 {
		return nil, nil, nil, fmt.Errorf("invalid decode dims hidden=%d vocab=%d", cfg.HiddenSize, cfg.VocabSize)
	}
	if m.Norm == nil || (m.LMHead == nil && m.LMHeadGGUF == nil) {
		return nil, nil, nil, fmt.Errorf("model final norm or LM head is not loaded")
	}
	h, vocab := cfg.HiddenSize, cfg.VocabSize
	flatHiddenLen, okH := checkedProduct(B, h)
	flatLogitsLen, okL := checkedProduct(B, vocab)
	lmLen, okLM := checkedProduct(vocab, h)
	if !okH || !okL || !okLM {
		return nil, nil, nil, fmt.Errorf("decode batch dimension overflow B=%d hidden=%d vocab=%d", B, h, vocab)
	}
	norm := m.Norm.Data()
	if len(norm) < h {
		return nil, nil, nil, fmt.Errorf("final norm len=%d, want at least %d", len(norm), h)
	}
	flatHidden := make([]float32, flatHiddenLen)
	finalActivations = make([][]float32, B)
	for i, row := range hiddenRows {
		if len(row) != h {
			return nil, nil, nil, fmt.Errorf("hidden row %d len=%d, want %d", i, len(row), h)
		}
		dst := flatHidden[i*h : (i+1)*h]
		copy(dst, row)
		if cfg.ModelType == "gemma3_text" {
			simd.RMSNormBF16(dst, norm, float32(cfg.RMSNormEps))
		} else {
			rmsNormInPlace(dst, norm, float32(cfg.RMSNormEps))
		}
		finalActivations[i] = append([]float32(nil), dst...)
	}
	flatLogits := make([]float32, flatLogitsLen)
	softcapNeeded := false
	if m.LMHead != nil {
		lmData := m.LMHead.Data()
		if len(lmData) != lmLen {
			return nil, nil, nil, fmt.Errorf("LM head data len=%d, want %d", len(lmData), lmLen)
		}
		if simd.SgemmNTTo(flatLogits, flatHidden, lmData, B, vocab, h, 1.0, h, h, vocab) {
			softcapNeeded = true
			goto logitsDone
		}
	}
	for i := 0; i < B; i++ {
		if err := m.LMHeadLogitsInto(flatLogits[i*vocab:(i+1)*vocab], flatHidden[i*h:(i+1)*h]); err != nil {
			return nil, nil, nil, err
		}
	}
logitsDone:
	if softcapNeeded {
		applyLlamaFinalLogitSoftcap(flatLogits, cfg.FinalLogitSoftcapping)
		for i := 0; i < B; i++ {
			applyLlamaSuppressTokens(flatLogits[i*vocab:(i+1)*vocab], m.SuppressTokens)
		}
	}
	logitsRows = make([][]float32, B)
	tokens = make([]int, B)
	for i := 0; i < B; i++ {
		row := append([]float32(nil), flatLogits[i*vocab:(i+1)*vocab]...)
		logitsRows[i] = row
		tok, _, err := ArgmaxLogits(row)
		if err != nil {
			return nil, nil, nil, err
		}
		tokens[i] = tok
	}
	return finalActivations, logitsRows, tokens, nil
}

// finishCPUDecodeStep is kept as the internal spelling used by existing decode
// paths; external orchestration layers should call FinishCPUDecodeStep.
func (m *LlamaModel) finishCPUDecodeStep(hidden []float32) (finalActivation []float32, logits []float32, token int, err error) {
	return m.FinishCPUDecodeStep(hidden)
}
