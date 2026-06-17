package model

import (
	"fmt"
	"math"

	"github.com/rcarmo/go-pherence/backends/mlx"
)

// MTPDrafterState carries the hidden-state-conditioned drafter inputs between
// speculative iterations. PreviousToken is the last token emitted/accepted by
// the verifier path; Activation is the main-model-width verifier activation
// used together with that token's embedding for the next drafter step.
type MTPDrafterState struct {
	PreviousToken int
	Activation    []float32
}

// MTPDrafterStepResult is one drafter iteration: the drafted token plus the
// projected main-model-width activation to carry into the next drafter step.
type MTPDrafterStepResult struct {
	Token          int
	Logits         []float32
	NextActivation []float32
	NextState      MTPDrafterState
}

// MTPDrafterExternalKV is the read-only main-model KV view consumed by q-only
// drafter layers. SourceLayers maps drafter layer index -> main-model KV layer.
// K/V are flat per-source-layer sequences with element width equal to the
// drafter layer's KV head count times head dim.
type MTPDrafterExternalKV struct {
	K            [][]float32
	V            [][]float32
	SourceLayers []int
	SeqLen       int
}

// NewMTPDrafterExternalKV builds the default one-to-one q-only source mapping
// for a Gemma4 MTP drafter: drafter layer i reads external K/V source i. This
// is the native local assistant layout; callers with packed/shared source views
// can still construct MTPDrafterExternalKV manually.
func NewMTPDrafterExternalKV(d *Gemma4MTPDrafter, k, v [][]float32, seqLen int) (*MTPDrafterExternalKV, error) {
	if d == nil {
		return nil, fmt.Errorf("nil drafter")
	}
	if d.Config.NumLayers < 0 {
		return nil, fmt.Errorf("invalid drafter layer count %d", d.Config.NumLayers)
	}
	sources := make([]int, d.Config.NumLayers)
	for i := range sources {
		sources[i] = i
	}
	externalKV := &MTPDrafterExternalKV{K: k, V: v, SourceLayers: sources, SeqLen: seqLen}
	if d.Config.NumLayers > 0 {
		if err := validateMTPDrafterExternalKV(d, externalKV); err != nil {
			return nil, err
		}
	}
	return externalKV, nil
}

// NewMTPDrafterState validates and copies the activation carry for one drafter
// loop. The activation width is the main/backbone model hidden size, not the
// assistant hidden size.
func NewMTPDrafterState(previousToken int, activation []float32, backboneHiddenSize int) (MTPDrafterState, error) {
	if previousToken < 0 {
		return MTPDrafterState{}, fmt.Errorf("previous token %d out of range", previousToken)
	}
	if backboneHiddenSize <= 0 {
		return MTPDrafterState{}, fmt.Errorf("invalid backbone hidden size %d", backboneHiddenSize)
	}
	if len(activation) != backboneHiddenSize {
		return MTPDrafterState{}, fmt.Errorf("activation len=%d, want %d", len(activation), backboneHiddenSize)
	}
	return MTPDrafterState{PreviousToken: previousToken, Activation: append([]float32(nil), activation...)}, nil
}

// RunMTPDrafterStep is one hidden-state-conditioned drafter iteration. It is a
// convenience wrapper for projection-only/zero-layer drafter fixtures; q-only
// drafter layers require RunMTPDrafterStepWithExternalKV so their external
// main-model K/V view is explicit.
func (m *LlamaModel) RunMTPDrafterStep(d *Gemma4MTPDrafter, state MTPDrafterState) (MTPDrafterStepResult, error) {
	return m.RunMTPDrafterStepWithExternalKV(d, state, nil)
}

// RunMTPDrafterStepWithExternalKV is RunMTPDrafterStep plus the explicit
// external/main-model KV view needed by q-only drafter layers.
func (m *LlamaModel) RunMTPDrafterStepWithExternalKV(d *Gemma4MTPDrafter, state MTPDrafterState, externalKV *MTPDrafterExternalKV) (MTPDrafterStepResult, error) {
	if err := m.validateMTPDrafterStepModelFull(d, state, externalKV); err != nil {
		return MTPDrafterStepResult{}, err
	}
	backboneEmbedding := make([]float32, d.BackboneHiddenSize)
	if err := m.TokenEmbeddingInto(backboneEmbedding, state.PreviousToken); err != nil {
		return MTPDrafterStepResult{}, fmt.Errorf("drafter backbone embedding: %w", err)
	}
	if m.Config.ModelType == "gemma3_text" || m.Config.ModelType == "gemma4_text" {
		scale := float32(math.Sqrt(float64(m.Config.HiddenSize)))
		for i := range backboneEmbedding {
			backboneEmbedding[i] *= scale
		}
	}
	assistantHidden := make([]float32, d.Config.HiddenSize)
	if err := d.PreProjectInto(assistantHidden, backboneEmbedding, state.Activation); err != nil {
		return MTPDrafterStepResult{}, err
	}
	for l := 0; l < d.Config.NumLayers; l++ {
		var err error
		assistantHidden, err = runMTPDrafterQOnlyLayer(m, d, assistantHidden, l, externalKV)
		if err != nil {
			return MTPDrafterStepResult{}, err
		}
	}
	if d.Norm != nil {
		norm := d.Norm.Data()
		if len(norm) < d.Config.HiddenSize {
			return MTPDrafterStepResult{}, fmt.Errorf("drafter final norm len=%d, want at least %d", len(norm), d.Config.HiddenSize)
		}
		drafterRMSNormInPlace(d, assistantHidden, norm)
	}
	nextActivation := make([]float32, d.BackboneHiddenSize)
	if err := d.PostProjectInto(nextActivation, assistantHidden); err != nil {
		return MTPDrafterStepResult{}, err
	}
	logits := make([]float32, d.Config.VocabSize)
	if d.EmbedTokens != nil || d.EmbedTokensMLX != nil {
		if err := d.AssistantLogitsInto(logits, assistantHidden); err != nil {
			return MTPDrafterStepResult{}, err
		}
	} else if err := m.LMHeadLogitsInto(logits, nextActivation); err != nil {
		return MTPDrafterStepResult{}, err
	}
	tok, _, err := ArgmaxLogits(logits)
	if err != nil {
		return MTPDrafterStepResult{}, err
	}
	nextState, err := NewMTPDrafterState(tok, nextActivation, d.BackboneHiddenSize)
	if err != nil {
		return MTPDrafterStepResult{}, err
	}
	return MTPDrafterStepResult{Token: tok, Logits: logits, NextActivation: append([]float32(nil), nextActivation...), NextState: nextState}, nil
}

func (m *LlamaModel) validateMTPDrafterStepModel(d *Gemma4MTPDrafter, state MTPDrafterState) error {
	if m == nil {
		return fmt.Errorf("nil model")
	}
	if d == nil {
		return fmt.Errorf("nil drafter")
	}
	if d.Config.HiddenSize <= 0 || d.BackboneHiddenSize <= 0 || d.Config.VocabSize <= 0 {
		return fmt.Errorf("invalid drafter dims hidden=%d backbone=%d vocab=%d", d.Config.HiddenSize, d.BackboneHiddenSize, d.Config.VocabSize)
	}
	if m.Config.HiddenSize != d.BackboneHiddenSize || m.Config.VocabSize != d.Config.VocabSize {
		return fmt.Errorf("model/drafter dims mismatch model h/vocab=%d/%d drafter backbone/vocab=%d/%d", m.Config.HiddenSize, m.Config.VocabSize, d.BackboneHiddenSize, d.Config.VocabSize)
	}
	if state.PreviousToken < 0 || state.PreviousToken >= d.Config.VocabSize {
		return fmt.Errorf("previous token %d out of range [0,%d)", state.PreviousToken, d.Config.VocabSize)
	}
	if len(state.Activation) != d.BackboneHiddenSize {
		return fmt.Errorf("state activation len=%d, want %d", len(state.Activation), d.BackboneHiddenSize)
	}
	if (len(d.PreProjection) == 0 && d.PreProjectionMLX == nil) || (len(d.PostProjection) == 0 && d.PostProjectionMLX == nil) {
		return fmt.Errorf("drafter projection weights are not loaded")
	}
	preWidth, ok := checkedProduct(2, d.BackboneHiddenSize)
	if !ok {
		return fmt.Errorf("drafter pre_projection width overflows for backbone=%d", d.BackboneHiddenSize)
	}
	if d.PreProjectionMLX == nil {
		wantPre, ok := checkedProduct(d.Config.HiddenSize, preWidth)
		if !ok {
			return fmt.Errorf("drafter pre_projection size overflows hidden=%d backbone=%d", d.Config.HiddenSize, d.BackboneHiddenSize)
		}
		if len(d.PreProjection) < wantPre {
			return fmt.Errorf("drafter pre_projection len=%d, want at least %d", len(d.PreProjection), wantPre)
		}
	} else if err := validateDrafterMLXWeight(-1, "pre_projection", d.PreProjectionMLX, d.Config.HiddenSize, preWidth); err != nil {
		return err
	}
	if d.PostProjectionMLX == nil {
		wantPost, ok := checkedProduct(d.BackboneHiddenSize, d.Config.HiddenSize)
		if !ok {
			return fmt.Errorf("drafter post_projection size overflows hidden=%d backbone=%d", d.Config.HiddenSize, d.BackboneHiddenSize)
		}
		if len(d.PostProjection) < wantPost {
			return fmt.Errorf("drafter post_projection len=%d, want at least %d", len(d.PostProjection), wantPost)
		}
	} else if err := validateDrafterMLXWeight(-1, "post_projection", d.PostProjectionMLX, d.BackboneHiddenSize, d.Config.HiddenSize); err != nil {
		return err
	}
	return m.validateMTPDrafterStepModelShell(d, state)
}

func (m *LlamaModel) validateMTPDrafterStepModelShell(d *Gemma4MTPDrafter, state MTPDrafterState) error {
	if d.Config.NumLayers != len(d.Layers) {
		return fmt.Errorf("drafter layer count=%d, want %d", len(d.Layers), d.Config.NumLayers)
	}
	if d.Config.NumLayers == 0 {
		return nil
	}
	if d.Norm == nil || len(d.Norm.Data()) < d.Config.HiddenSize {
		return fmt.Errorf("drafter final norm is not loaded or too small")
	}
	return nil
}

func (m *LlamaModel) validateMTPDrafterStepModelFull(d *Gemma4MTPDrafter, state MTPDrafterState, externalKV *MTPDrafterExternalKV) error {
	if err := m.validateMTPDrafterStepModel(d, state); err != nil {
		return err
	}
	if d.Config.NumLayers == 0 {
		return nil
	}
	return validateMTPDrafterExternalKV(d, externalKV)
}

func runMTPDrafterQOnlyLayer(m *LlamaModel, d *Gemma4MTPDrafter, hidden []float32, layerIdx int, externalKV *MTPDrafterExternalKV) ([]float32, error) {
	if d == nil || layerIdx < 0 || layerIdx >= len(d.Layers) {
		return nil, fmt.Errorf("invalid drafter layer %d", layerIdx)
	}
	layer := &d.Layers[layerIdx]
	h := d.Config.HiddenSize
	headDim := drafterLayerHeadDim(d, layerIdx)
	qDim, ok := checkedProduct(d.Config.NumHeads, headDim)
	if headDim <= 0 || !ok {
		return nil, fmt.Errorf("invalid drafter layer %d q dim heads=%d headDim=%d", layerIdx, d.Config.NumHeads, headDim)
	}
	source := externalKV.SourceLayers[layerIdx]
	residual := append([]float32(nil), hidden...)
	normed := append([]float32(nil), hidden...)
	drafterRMSNormInPlace(d, normed, layer.InputNorm.Data())
	q := make([]float32, qDim)
	if layer.QWm != nil {
		if !mlx.GemvTo(q, normed, layer.QWm) {
			return nil, fmt.Errorf("drafter layer %d Q MLX projection failed", layerIdx)
		}
	} else if len(layer.QWBF16) >= qDim*h {
		if !gemvBF16BF16(q, normed, layer.QWBF16, h, qDim) {
			return nil, fmt.Errorf("drafter layer %d Q BF16 projection failed", layerIdx)
		}
	} else {
		gemvNT(q, normed, layer.QW, h, qDim)
	}
	qNorm := layer.QNorm.Data()
	for head := 0; head < d.Config.NumHeads; head++ {
		drafterRMSNormInPlace(d, q[head*headDim:(head+1)*headDim], qNorm)
	}
	if d.Config.ModelType == "gemma4_text" {
		pos := externalKV.SeqLen
		isSWA := true
		if layerIdx >= 0 && layerIdx < len(d.Config.LayerTypes) {
			isSWA = d.Config.LayerTypes[layerIdx] != "full_attention"
		}
		if isSWA {
			rotHalf := headDim / 2
			freqs := []float32(nil)
			if len(d.RopeFreqsSWA) > 0 && d.RopeHalfSWA > 0 {
				rotHalf = d.RopeHalfSWA
				freqs = d.RopeFreqsSWA
			} else if m != nil && len(m.RopeFreqsSWA) > 0 && m.RopeHalfSWA > 0 {
				rotHalf = m.RopeHalfSWA
				freqs = m.RopeFreqsSWA
			} else {
				freqs = buildRoPEFreqs(pos+1, rotHalf, headDim, 10000)
			}
			applyRoPEPartial(q, freqs, pos, d.Config.NumHeads, headDim, rotHalf)
		} else {
			rotHalf := int(float64(headDim)*0.25) / 2
			freqs := []float32(nil)
			if len(d.RopeFreqsFull) > 0 && d.RopeHalfFull > 0 {
				rotHalf = d.RopeHalfFull
				freqs = d.RopeFreqsFull
			} else if m != nil && len(m.RopeFreqsFull) > 0 && m.RopeHalfFull > 0 {
				rotHalf = m.RopeHalfFull
				freqs = m.RopeFreqsFull
			} else {
				freqs = buildRoPEFreqs(pos+1, rotHalf, headDim, 1000000)
			}
			applyRoPEPartial(q, freqs, pos, d.Config.NumHeads, headDim, rotHalf)
		}
	}
	kvHeads := drafterLayerKVHeads(d, layerIdx)
	attnOut := drafterGQAAttention(d, q, externalKV.K[source], externalKV.V[source], externalKV.SeqLen, d.Config.NumHeads, kvHeads, headDim)
	if attnOut == nil {
		return nil, fmt.Errorf("drafter layer %d external attention failed", layerIdx)
	}
	oOut := make([]float32, h)
	if layer.OWm != nil {
		if !mlx.GemvTo(oOut, attnOut, layer.OWm) {
			return nil, fmt.Errorf("drafter layer %d O MLX projection failed", layerIdx)
		}
	} else if len(layer.OWBF16) >= h*qDim {
		if !gemvBF16BF16(oOut, attnOut, layer.OWBF16, qDim, h) {
			return nil, fmt.Errorf("drafter layer %d O BF16 projection failed", layerIdx)
		}
	} else {
		gemvNT(oOut, attnOut, layer.OW, qDim, h)
	}
	if layer.PreFFNNorm != nil {
		drafterRMSNormInPlace(d, oOut, layer.PostNorm.Data())
		for i := 0; i < h; i++ {
			hidden[i] = residual[i] + oOut[i]
		}
		copy(residual, hidden)
	} else {
		for i := 0; i < h; i++ {
			hidden[i] = residual[i] + oOut[i]
		}
		copy(residual, hidden)
		drafterRMSNormInPlace(d, hidden, layer.PostNorm.Data())
	}
	mlpInput := hidden
	if layer.PreFFNNorm != nil {
		mlpInput = append([]float32(nil), hidden...)
		drafterRMSNormInPlace(d, mlpInput, layer.PreFFNNorm.Data())
	}
	gate := make([]float32, d.Config.Intermediate)
	up := make([]float32, d.Config.Intermediate)
	if layer.GateWm != nil {
		if !mlx.GemvTo(gate, mlpInput, layer.GateWm) {
			return nil, fmt.Errorf("drafter layer %d gate MLX projection failed", layerIdx)
		}
	} else if len(layer.GateWBF16) >= d.Config.Intermediate*h {
		if !gemvBF16BF16(gate, mlpInput, layer.GateWBF16, h, d.Config.Intermediate) {
			return nil, fmt.Errorf("drafter layer %d gate BF16 projection failed", layerIdx)
		}
	} else {
		gemvNT(gate, mlpInput, layer.GateW, h, d.Config.Intermediate)
	}
	if layer.UpWm != nil {
		if !mlx.GemvTo(up, mlpInput, layer.UpWm) {
			return nil, fmt.Errorf("drafter layer %d up MLX projection failed", layerIdx)
		}
	} else if len(layer.UpWBF16) >= d.Config.Intermediate*h {
		if !gemvBF16BF16(up, mlpInput, layer.UpWBF16, h, d.Config.Intermediate) {
			return nil, fmt.Errorf("drafter layer %d up BF16 projection failed", layerIdx)
		}
	} else {
		gemvNT(up, mlpInput, layer.UpW, h, d.Config.Intermediate)
	}
	if d.Config.ModelType == "gemma4_text" {
		ggmlGELUMulInPlace(gate, up)
	} else {
		for i := range gate {
			gate[i] = geluTanh(gate[i]) * up[i]
		}
	}
	down := make([]float32, h)
	if layer.DownWm != nil {
		if !mlx.GemvTo(down, gate, layer.DownWm) {
			return nil, fmt.Errorf("drafter layer %d down MLX projection failed", layerIdx)
		}
	} else if len(layer.DownWBF16) >= h*d.Config.Intermediate {
		if !gemvBF16BF16(down, gate, layer.DownWBF16, d.Config.Intermediate, h) {
			return nil, fmt.Errorf("drafter layer %d down BF16 projection failed", layerIdx)
		}
	} else {
		gemvNT(down, gate, layer.DownW, d.Config.Intermediate, h)
	}
	if layer.PostFFNNorm != nil {
		drafterRMSNormInPlace(d, down, layer.PostFFNNorm.Data())
	}
	for i := 0; i < h; i++ {
		hidden[i] = residual[i] + down[i]
	}
	if layer.LayerScalar != 1 {
		for i := range hidden {
			hidden[i] *= layer.LayerScalar
		}
	}
	return hidden, nil
}

// LayerHeadDim returns the effective per-head width for a drafter layer.
// Gemma4 full-attention layers use GlobalHeadDim when HeadDimLocal is absent,
// matching the main-model LayerHeadDim contract used by verifier paths.
func (d *Gemma4MTPDrafter) LayerHeadDim(layerIdx int) int {
	if d == nil {
		return 0
	}
	if layerIdx >= 0 && layerIdx < len(d.Layers) && d.Layers[layerIdx].HeadDimLocal > 0 {
		return d.Layers[layerIdx].HeadDimLocal
	}
	if layerIdx >= 0 && layerIdx < len(d.Config.LayerTypes) && d.Config.LayerTypes[layerIdx] == "full_attention" && d.Config.GlobalHeadDim > 0 {
		return d.Config.GlobalHeadDim
	}
	return d.Config.HeadDim
}

// LayerKVHeads returns the effective K/V head count for a drafter layer.
func (d *Gemma4MTPDrafter) LayerKVHeads(layerIdx int) int {
	if d == nil {
		return 0
	}
	if layerIdx >= 0 && layerIdx < len(d.Config.LayerTypes) && d.Config.LayerTypes[layerIdx] == "full_attention" && d.Config.NumGlobalKVHeads > 0 {
		return d.Config.NumGlobalKVHeads
	}
	return d.Config.NumKVHeads
}

// LayerKVDim returns the flat per-token external K/V width consumed by a
// q-only drafter layer.
func (d *Gemma4MTPDrafter) LayerKVDim(layerIdx int) (int, error) {
	if d == nil {
		return 0, fmt.Errorf("nil drafter")
	}
	if layerIdx < 0 || layerIdx >= d.Config.NumLayers || layerIdx >= len(d.Layers) {
		return 0, fmt.Errorf("drafter layer %d out of range", layerIdx)
	}
	headDim := d.LayerHeadDim(layerIdx)
	kvHeads := d.LayerKVHeads(layerIdx)
	kvDim, ok := checkedProduct(kvHeads, headDim)
	if headDim <= 0 || kvHeads <= 0 || !ok {
		return 0, fmt.Errorf("invalid drafter layer %d KV dim kvHeads=%d headDim=%d", layerIdx, kvHeads, headDim)
	}
	return kvDim, nil
}

func drafterLayerHeadDim(d *Gemma4MTPDrafter, layerIdx int) int {
	return d.LayerHeadDim(layerIdx)
}

func drafterLayerKVHeads(d *Gemma4MTPDrafter, layerIdx int) int {
	return d.LayerKVHeads(layerIdx)
}

func drafterGQAAttention(d *Gemma4MTPDrafter, q, kCache, vCache []float32, seqLen, numHeads, numKVHeads, headDim int) []float32 {
	if d != nil && d.Config.ModelType == "gemma4_text" {
		return gqaAttentionScale(q, kCache, vCache, seqLen, numHeads, numKVHeads, headDim, 1.0)
	}
	return gqaAttention(q, kCache, vCache, seqLen, numHeads, numKVHeads, headDim)
}

func drafterRMSNormInPlace(d *Gemma4MTPDrafter, x, weight []float32) {
	if d != nil && d.Config.ModelType == "gemma3_text" {
		rmsNormBF16(x, weight, float32(d.Config.RMSNormEps))
		return
	}
	rmsNormInPlace(x, weight, float32(d.Config.RMSNormEps))
}

func validateMTPDrafterExternalKV(d *Gemma4MTPDrafter, externalKV *MTPDrafterExternalKV) error {
	if externalKV == nil {
		return fmt.Errorf("MTP drafter external KV is required for q-only layers")
	}
	if externalKV.SeqLen <= 0 {
		return fmt.Errorf("invalid MTP drafter external KV seq len %d", externalKV.SeqLen)
	}
	if len(externalKV.SourceLayers) != d.Config.NumLayers {
		return fmt.Errorf("drafter external KV source layers=%d, want %d", len(externalKV.SourceLayers), d.Config.NumLayers)
	}
	if d.Config.NumHeads <= 0 || d.Config.NumKVHeads <= 0 || d.Config.HeadDim <= 0 || d.Config.Intermediate <= 0 {
		return fmt.Errorf("invalid drafter q-only dims heads=%d kvHeads=%d headDim=%d intermediate=%d", d.Config.NumHeads, d.Config.NumKVHeads, d.Config.HeadDim, d.Config.Intermediate)
	}
	allowDuplicateSources := d.Config.ModelType == "gemma4_text"
	usedSources := map[int]bool{}
	for i := range d.Layers {
		layer := &d.Layers[i]
		headDim := drafterLayerHeadDim(d, i)
		qDim, ok := checkedProduct(d.Config.NumHeads, headDim)
		if headDim <= 0 || !ok {
			return fmt.Errorf("invalid drafter layer %d q dim heads=%d headDim=%d", i, d.Config.NumHeads, headDim)
		}
		kvHeads := drafterLayerKVHeads(d, i)
		kvDim, ok := checkedProduct(kvHeads, headDim)
		if !ok {
			return fmt.Errorf("invalid drafter layer %d KV dim kvHeads=%d headDim=%d", i, d.Config.NumKVHeads, headDim)
		}
		if layer.KVSourceLayer != -1 {
			return fmt.Errorf("drafter layer %d has unexpected owned/shared KV source %d, want q-only -1", i, layer.KVSourceLayer)
		}
		source := externalKV.SourceLayers[i]
		if source < 0 || source >= len(externalKV.K) || source >= len(externalKV.V) {
			return fmt.Errorf("drafter layer %d external KV source %d out of range K/V=%d/%d", i, source, len(externalKV.K), len(externalKV.V))
		}
		if !allowDuplicateSources {
			if usedSources[source] {
				return fmt.Errorf("drafter layer %d reuses external KV source %d", i, source)
			}
			usedSources[source] = true
		}
		wantKV, ok := checkedProduct(externalKV.SeqLen, kvDim)
		if !ok {
			return fmt.Errorf("drafter layer %d external KV length overflows seq=%d kvDim=%d", i, externalKV.SeqLen, kvDim)
		}
		if len(externalKV.K[source]) != wantKV || len(externalKV.V[source]) != wantKV {
			return fmt.Errorf("drafter layer %d external KV K/V=%d/%d, want %d", i, len(externalKV.K[source]), len(externalKV.V[source]), wantKV)
		}
		if layer.InputNorm == nil || layer.PostNorm == nil || layer.PreFFNNorm == nil || layer.PostFFNNorm == nil || layer.QNorm == nil {
			return fmt.Errorf("drafter layer %d missing q-only norms", i)
		}
		if len(layer.InputNorm.Data()) < d.Config.HiddenSize || len(layer.PostNorm.Data()) < d.Config.HiddenSize || len(layer.PreFFNNorm.Data()) < d.Config.HiddenSize || len(layer.PostFFNNorm.Data()) < d.Config.HiddenSize || len(layer.QNorm.Data()) < headDim {
			return fmt.Errorf("drafter layer %d norm dims are too small", i)
		}
		qWeightLen, okQWeight := checkedProduct(qDim, d.Config.HiddenSize)
		oWeightLen, okOWeight := checkedProduct(d.Config.HiddenSize, qDim)
		if !okQWeight || !okOWeight {
			return fmt.Errorf("drafter layer %d attention weight dims overflow qDim=%d hidden=%d", i, qDim, d.Config.HiddenSize)
		}
		if (len(layer.QW) != qWeightLen && layer.QWm == nil) || (len(layer.OW) != oWeightLen && layer.OWm == nil) {
			return fmt.Errorf("drafter layer %d attention weight dims Q/O=%d/%d, want %d/%d", i, len(layer.QW), len(layer.OW), qWeightLen, oWeightLen)
		}
		if err := validateDrafterMLXWeight(i, "q_proj", layer.QWm, qDim, d.Config.HiddenSize); err != nil {
			return err
		}
		if err := validateDrafterMLXWeight(i, "o_proj", layer.OWm, d.Config.HiddenSize, qDim); err != nil {
			return err
		}
		gateWeightLen, okGateWeight := checkedProduct(d.Config.Intermediate, d.Config.HiddenSize)
		downWeightLen, okDownWeight := checkedProduct(d.Config.HiddenSize, d.Config.Intermediate)
		if !okGateWeight || !okDownWeight {
			return fmt.Errorf("drafter layer %d MLP weight dims overflow intermediate=%d hidden=%d", i, d.Config.Intermediate, d.Config.HiddenSize)
		}
		if (len(layer.GateW) != gateWeightLen && layer.GateWm == nil) || (len(layer.UpW) != gateWeightLen && layer.UpWm == nil) || (len(layer.DownW) != downWeightLen && layer.DownWm == nil) {
			return fmt.Errorf("drafter layer %d MLP weight dims are invalid", i)
		}
		if err := validateDrafterMLXWeight(i, "gate_proj", layer.GateWm, d.Config.Intermediate, d.Config.HiddenSize); err != nil {
			return err
		}
		if err := validateDrafterMLXWeight(i, "up_proj", layer.UpWm, d.Config.Intermediate, d.Config.HiddenSize); err != nil {
			return err
		}
		if err := validateDrafterMLXWeight(i, "down_proj", layer.DownWm, d.Config.HiddenSize, d.Config.Intermediate); err != nil {
			return err
		}
	}
	return nil
}

func validateDrafterMLXWeight(layerIdx int, name string, qw *mlx.QuantWeight, outDim, inDim int) error {
	if qw == nil {
		return nil
	}
	if qw.OutDim != outDim || qw.InDim != inDim {
		return fmt.Errorf("drafter layer %d %s MLX dims out/in=%d/%d, want %d/%d", layerIdx, name, qw.OutDim, qw.InDim, outDim, inDim)
	}
	if err := mlx.ValidateQuantWeight(qw); err != nil {
		return fmt.Errorf("drafter layer %d %s MLX weight: %w", layerIdx, name, err)
	}
	return nil
}
