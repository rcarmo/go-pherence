package lfm2

import (
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

// AttentionCPUState owns token-major K/V rows for one full-attention layer.
type AttentionCPUState struct {
	K []float32
	V []float32
}

func (s AttentionCPUState) Clone() AttentionCPUState {
	return AttentionCPUState{K: append([]float32(nil), s.K...), V: append([]float32(nil), s.V...)}
}

// FullAttentionCPU is one LFM2 GQA layer operator. It owns immutable projection
// and Q/K normalization weights; request state is passed explicitly.
type FullAttentionCPU struct {
	cfg     Config
	layer   int
	q, k, v lfm2Linear
	out     lfm2Linear
	qNorm   []float32
	kNorm   []float32
}

func LoadFullAttentionCPU(src Float32TensorSource, cfg Config, layer int) (*FullAttentionCPU, error) {
	if src == nil {
		return nil, fmt.Errorf("nil LFM2 attention tensor source")
	}
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if layer < 0 || layer >= len(cfg.LayerTypes) || cfg.LayerTypes[layer] != "full_attention" {
		return nil, fmt.Errorf("LFM2 layer %d is not a full-attention layer", layer)
	}
	queryWidth := sizeProduct(cfg.NumAttentionHeads, cfg.HeadDim)
	kvWidth := sizeProduct(cfg.NumKeyValueHeads, cfg.HeadDim)
	if queryWidth < 0 || kvWidth < 0 {
		return nil, fmt.Errorf("LFM2 attention projection width overflows")
	}
	prefix := fmt.Sprintf("model.layers.%d.self_attn", layer)
	m := &FullAttentionCPU{cfg: cfg, layer: layer}
	var err error
	if m.q, err = loadLFM2Linear(src, prefix+".q_proj", cfg.HiddenSize, queryWidth, false); err != nil {
		return nil, err
	}
	if m.k, err = loadLFM2Linear(src, prefix+".k_proj", cfg.HiddenSize, kvWidth, false); err != nil {
		return nil, err
	}
	if m.v, err = loadLFM2Linear(src, prefix+".v_proj", cfg.HiddenSize, kvWidth, false); err != nil {
		return nil, err
	}
	if m.out, err = loadLFM2Linear(src, prefix+".out_proj", queryWidth, cfg.HiddenSize, false); err != nil {
		return nil, err
	}
	if m.qNorm, err = loadLFM2Tensor(src, prefix+".q_layernorm.weight", []int{cfg.HeadDim}); err != nil {
		return nil, err
	}
	if m.kNorm, err = loadLFM2Tensor(src, prefix+".k_layernorm.weight", []int{cfg.HeadDim}); err != nil {
		return nil, err
	}
	return m, nil
}

// ForwardToken executes one causal GQA step and returns a new KV state. The
// caller's input and state remain unchanged on success and failure.
func (m *FullAttentionCPU) ForwardToken(input []float32, state AttentionCPUState, pos int) ([]float32, AttentionCPUState, error) {
	if m == nil {
		return nil, state, fmt.Errorf("nil LFM2 CPU attention")
	}
	maxPosition := m.cfg.MaxPositionEmbeddings
	if maxPosition == 0 {
		maxPosition = 128000
	}
	if pos < 0 || pos >= maxPosition {
		return nil, state, fmt.Errorf("invalid LFM2 attention position=%d max=%d", pos, maxPosition)
	}
	if len(input) != m.cfg.HiddenSize {
		return nil, state, fmt.Errorf("invalid LFM2 attention input=%d want %d", len(input), m.cfg.HiddenSize)
	}
	queryWidth := sizeProduct(m.cfg.NumAttentionHeads, m.cfg.HeadDim)
	kvWidth := sizeProduct(m.cfg.NumKeyValueHeads, m.cfg.HeadDim)
	wantState := sizeProduct(pos, kvWidth)
	if queryWidth < 0 || kvWidth < 0 || wantState < 0 || len(state.K) != wantState || len(state.V) != wantState {
		return nil, state, fmt.Errorf("invalid LFM2 attention state K/V=%d/%d want %d", len(state.K), len(state.V), wantState)
	}
	q := make([]float32, queryWidth)
	k := make([]float32, kvWidth)
	v := make([]float32, kvWidth)
	if err := m.q.forward(q, input); err != nil {
		return nil, state, err
	}
	if err := m.k.forward(k, input); err != nil {
		return nil, state, err
	}
	if err := m.v.forward(v, input); err != nil {
		return nil, state, err
	}
	if err := normLFM2Heads(q, m.qNorm, m.cfg.NumAttentionHeads, m.cfg.HeadDim, float32(m.cfg.NormEps)); err != nil {
		return nil, state, err
	}
	if err := normLFM2Heads(k, m.kNorm, m.cfg.NumKeyValueHeads, m.cfg.HeadDim, float32(m.cfg.NormEps)); err != nil {
		return nil, state, err
	}
	rope := simd.BuildRoPEFreqs(pos+1, m.cfg.HeadDim/2, m.cfg.HeadDim, m.cfg.RoPE.Theta)
	if !simd.ApplyRoPETo(q, rope, pos, m.cfg.NumAttentionHeads, m.cfg.HeadDim) || !simd.ApplyRoPETo(k, rope, pos, m.cfg.NumKeyValueHeads, m.cfg.HeadDim) {
		return nil, state, fmt.Errorf("LFM2 attention RoPE failed")
	}
	next := AttentionCPUState{K: make([]float32, len(state.K)+kvWidth), V: make([]float32, len(state.V)+kvWidth)}
	copy(next.K, state.K)
	copy(next.K[len(state.K):], k)
	copy(next.V, state.V)
	copy(next.V[len(state.V):], v)
	attended := make([]float32, queryWidth)
	scores := make([]float32, pos+1)
	scale := float32(1 / math.Sqrt(float64(m.cfg.HeadDim)))
	if !simd.GQAAttentionScaleTo(attended, scores, q, next.K, next.V, pos+1, m.cfg.NumAttentionHeads, m.cfg.NumKeyValueHeads, m.cfg.HeadDim, scale) {
		return nil, state, fmt.Errorf("LFM2 GQA attention failed")
	}
	output := make([]float32, m.cfg.HiddenSize)
	if err := m.out.forward(output, attended); err != nil {
		return nil, state, err
	}
	return output, next, nil
}

func normLFM2Heads(x, weight []float32, heads, headDim int, eps float32) error {
	if eps == 0 {
		eps = 1e-5
	}
	if heads <= 0 || headDim <= 0 || len(x) != heads*headDim || len(weight) != headDim {
		return fmt.Errorf("invalid LFM2 QK norm dims values=%d weight=%d heads=%d head_dim=%d", len(x), len(weight), heads, headDim)
	}
	for head := 0; head < heads; head++ {
		if !simd.RMSNormTo(x[head*headDim:(head+1)*headDim], weight, eps) {
			return fmt.Errorf("LFM2 QK norm head %d failed", head)
		}
	}
	return nil
}
