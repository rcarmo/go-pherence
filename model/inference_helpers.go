package model

import (
	"fmt"
	"math"

	"github.com/rcarmo/go-pherence/backends/simd/runtime"
)

// TokenEmbeddingInto copies the raw token embedding row into dst.
func (m *LlamaModel) TokenEmbeddingInto(dst []float32, tokenID int) error {
	if m == nil || (m.EmbedTokens == nil && m.EmbedTokensGGUF == nil) {
		return fmt.Errorf("model embeddings are not loaded")
	}
	h := m.Config.HiddenSize
	if h <= 0 || m.Config.VocabSize <= 0 {
		return fmt.Errorf("invalid embedding config hidden=%d vocab=%d", h, m.Config.VocabSize)
	}
	if len(dst) != h {
		return fmt.Errorf("token embedding dst len=%d, want %d", len(dst), h)
	}
	if tokenID < 0 || tokenID >= m.Config.VocabSize {
		return fmt.Errorf("token id %d out of range [0,%d)", tokenID, m.Config.VocabSize)
	}
	if m.EmbedTokensGGUF != nil {
		if m.EmbedTokensGGUF.InDim != h || m.EmbedTokensGGUF.OutDim != m.Config.VocabSize {
			return fmt.Errorf("GGUF embedding dims out/in=%d/%d, want %d/%d", m.EmbedTokensGGUF.OutDim, m.EmbedTokensGGUF.InDim, m.Config.VocabSize, h)
		}
		return m.EmbedTokensGGUF.DequantRowTo(dst, tokenID)
	}
	emb := m.EmbedTokens.Data()
	start, ok := checkedProduct(tokenID, h)
	if !ok {
		return fmt.Errorf("embedding offset overflows for token=%d hidden=%d", tokenID, h)
	}
	need, ok := checkedProduct(tokenID+1, h)
	if !ok || len(emb) < need {
		return fmt.Errorf("embedding data len=%d, want at least %d", len(emb), need)
	}
	copy(dst, emb[start:need])
	return nil
}

// ScaledTokenEmbeddingInto copies the token embedding row and applies the same
// model-specific decode-time embedding scaling used by Generate.
func (m *LlamaModel) ScaledTokenEmbeddingInto(dst []float32, tokenID int) error {
	if err := m.TokenEmbeddingInto(dst, tokenID); err != nil {
		return err
	}
	if m.Config.ModelType == "gemma3_text" || m.Config.ModelType == "gemma4_text" {
		scale := float32(math.Sqrt(float64(m.Config.HiddenSize)))
		for i := range dst {
			dst[i] *= scale
		}
		// Gemma3 decode path historically matches BF16-scaled embeddings, but
		// llama.cpp Gemma4 keeps the scaled embedding row in F32 at layer 0.
		if m.Config.ModelType == "gemma3_text" {
			simd.ToBF16(dst)
		}
	}
	return nil
}

// Gemma4PerLayerInputs builds Gemma4 per-layer input slices for one token.
// Returned per-layer slices share a single backing buffer and remain valid as
// long as the returned [][]float32 is kept alive.
func (m *LlamaModel) Gemma4PerLayerInputs(hidden []float32, tokenID int) ([][]float32, error) {
	return m.Gemma4PerLayerInputsInto(nil, nil, hidden, tokenID)
}

// Gemma4PerLayerInputsInto is like Gemma4PerLayerInputs but writes into the
// caller-provided projBuf (len totalDim) and slices header (len NumLayers),
// avoiding a per-token allocation in the decode loop. Pass nil buffers to have
// fresh ones allocated.
func (m *LlamaModel) Gemma4PerLayerInputsInto(projBuf []float32, slices [][]float32, hidden []float32, tokenID int) ([][]float32, error) {
	if m == nil {
		return nil, fmt.Errorf("nil model")
	}
	cfg := m.Config
	if m.PerLayerModelProj == nil || cfg.HiddenPerLayer == 0 {
		return nil, nil
	}
	h := cfg.HiddenSize
	hpl := cfg.HiddenPerLayer
	nl := cfg.NumLayers
	if h <= 0 || hpl <= 0 || nl < 0 {
		return nil, fmt.Errorf("invalid per-layer input config hidden=%d hiddenPerLayer=%d layers=%d", h, hpl, nl)
	}
	totalDim, ok := checkedProduct(nl, hpl)
	if !ok {
		return nil, fmt.Errorf("per-layer input dimension overflow")
	}
	if len(hidden) != h {
		return nil, fmt.Errorf("per-layer input hidden len=%d, want %d", len(hidden), h)
	}
	if tokenID < 0 {
		return nil, fmt.Errorf("token id %d out of range", tokenID)
	}
	wantProj, ok := checkedProduct(totalDim, h)
	if !ok {
		return nil, fmt.Errorf("per-layer projection dimension overflow")
	}
	if len(m.PerLayerModelProj) != wantProj {
		return nil, fmt.Errorf("per-layer model projection len=%d, want %d", len(m.PerLayerModelProj), wantProj)
	}
	if len(m.PerLayerProjNorm) != hpl {
		return nil, fmt.Errorf("per-layer projection norm len=%d, want %d", len(m.PerLayerProjNorm), hpl)
	}
	if m.EmbedPerLayer != nil && tokenID < cfg.VocabPerLayer {
		need, ok := checkedProduct(cfg.VocabPerLayer, totalDim)
		if !ok {
			return nil, fmt.Errorf("per-layer embedding dimension overflow")
		}
		if len(m.EmbedPerLayer) < need {
			return nil, fmt.Errorf("per-layer embedding len=%d, want at least %d", len(m.EmbedPerLayer), need)
		}
	}
	if m.EmbedPerLayerGGUF != nil && tokenID < cfg.VocabPerLayer {
		if m.EmbedPerLayerGGUF.InDim != totalDim || m.EmbedPerLayerGGUF.OutDim != cfg.VocabPerLayer {
			return nil, fmt.Errorf("GGUF per-layer embedding dims out/in=%d/%d, want %d/%d", m.EmbedPerLayerGGUF.OutDim, m.EmbedPerLayerGGUF.InDim, cfg.VocabPerLayer, totalDim)
		}
	}

	var proj []float32
	if cap(projBuf) >= totalDim {
		proj = projBuf[:totalDim]
	} else {
		proj = make([]float32, totalDim)
	}
	gemvNT(proj, hidden, m.PerLayerModelProj, h, totalDim)
	for i := range proj {
		proj[i] *= m.PerLayerProjScale
	}
	for l := 0; l < nl; l++ {
		sl := proj[l*hpl : (l+1)*hpl]
		rmsNormInPlace(sl, m.PerLayerProjNorm, float32(cfg.RMSNormEps))
		traceMTPSummary("per_layer_proj", -1, l, tokenID, sl)
	}
	if m.EmbedPerLayerGGUF != nil && tokenID < cfg.VocabPerLayer {
		embRow := make([]float32, totalDim)
		if err := m.EmbedPerLayerGGUF.DequantRowTo(embRow, tokenID); err != nil {
			return nil, err
		}
		for l := 0; l < nl; l++ {
			selected := append([]float32(nil), embRow[l*hpl:(l+1)*hpl]...)
			for i := range selected {
				selected[i] *= m.EmbedPerLayerScale
			}
			traceMTPSummary("inp_per_layer_selected", -1, l, tokenID, selected)
		}
		for i := range proj {
			proj[i] = (proj[i] + embRow[i]*m.EmbedPerLayerScale) * m.PerLayerInputScale
		}
	} else if m.EmbedPerLayer != nil && tokenID < cfg.VocabPerLayer {
		embRow := m.EmbedPerLayer[tokenID*totalDim : (tokenID+1)*totalDim]
		for l := 0; l < nl; l++ {
			selected := append([]float32(nil), embRow[l*hpl:(l+1)*hpl]...)
			for i := range selected {
				selected[i] *= m.EmbedPerLayerScale
			}
			traceMTPSummary("inp_per_layer_selected", -1, l, tokenID, selected)
		}
		for i := range proj {
			proj[i] = (proj[i] + embRow[i]*m.EmbedPerLayerScale) * m.PerLayerInputScale
		}
	}

	var perLayerInputs [][]float32
	if cap(slices) >= nl {
		perLayerInputs = slices[:nl]
	} else {
		perLayerInputs = make([][]float32, nl)
	}
	for l := 0; l < nl; l++ {
		perLayerInputs[l] = proj[l*hpl : (l+1)*hpl]
	}
	return perLayerInputs, nil
}

// LMHeadLogitsInto computes logits = hidden · lm_head^T.
func (m *LlamaModel) LMHeadLogitsInto(logits, hidden []float32) error {
	if m == nil || (m.LMHead == nil && m.LMHeadGGUF == nil) {
		return fmt.Errorf("model LM head is not loaded")
	}
	h := m.Config.HiddenSize
	vocab := m.Config.VocabSize
	if h <= 0 || vocab <= 0 {
		return fmt.Errorf("invalid LM head config hidden=%d vocab=%d", h, vocab)
	}
	want, ok := checkedProduct(vocab, h)
	if !ok {
		return fmt.Errorf("LM head dimension overflow")
	}
	if len(hidden) != h {
		return fmt.Errorf("hidden len=%d, want %d", len(hidden), h)
	}
	if len(logits) != vocab {
		return fmt.Errorf("logits len=%d, want %d", len(logits), vocab)
	}
	if m.LMHeadGGUF != nil {
		if m.LMHeadGGUF.InDim != h || m.LMHeadGGUF.OutDim != vocab {
			return fmt.Errorf("GGUF LM head dims out/in=%d/%d, want %d/%d", m.LMHeadGGUF.OutDim, m.LMHeadGGUF.InDim, vocab, h)
		}
		if err := gemvGGUFQuantRows(logits, hidden, m.LMHeadGGUF, h, vocab); err != nil {
			return err
		}
		applyLlamaFinalLogitSoftcap(logits, m.Config.FinalLogitSoftcapping)
		applyLlamaSuppressTokens(logits, m.SuppressTokens)
		return nil
	}
	lmData := m.LMHead.Data()
	if len(lmData) != want {
		return fmt.Errorf("LM head data len=%d, want %d", len(lmData), want)
	}
	// logits[v] = hidden · lmData[v]; parallelized across the (large) vocab
	// dimension. Numerically identical to the per-row dot fallback below.
	if !simd.GemvRowsParallel(logits, hidden, lmData, vocab, h) {
		for v := 0; v < vocab; v++ {
			logits[v] = simdDot(hidden, lmData[v*h:(v+1)*h])
		}
	}
	applyLlamaFinalLogitSoftcap(logits, m.Config.FinalLogitSoftcapping)
	applyLlamaSuppressTokens(logits, m.SuppressTokens)
	return nil
}

func applyLlamaSuppressTokens(logits []float32, ids []int) {
	for _, id := range ids {
		if id >= 0 && id < len(logits) {
			logits[id] = float32(math.Inf(-1))
		}
	}
}

func applyLlamaFinalLogitSoftcap(logits []float32, c float64) {
	if c <= 0 {
		return
	}
	capF := float32(c)
	inv := float32(1.0 / c)
	for i, v := range logits {
		scaled := float32(v * inv)
		logits[i] = float32(math.Tanh(float64(scaled))) * capF
	}
}

// ArgmaxLogits returns the index and value of the maximum logit.
func ArgmaxLogits(logits []float32) (int, float32, error) {
	if len(logits) == 0 {
		return 0, 0, fmt.Errorf("empty logits")
	}
	maxIdx := 0
	maxVal := logits[0]
	for i, v := range logits[1:] {
		if v > maxVal {
			maxVal = v
			maxIdx = i + 1
		}
	}
	return maxIdx, maxVal, nil
}
