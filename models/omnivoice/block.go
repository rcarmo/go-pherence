// Package omnivoice implements the native CPU numerical core of OmniVoice.
// It is not yet a waveform synthesizer: sampling and the learned audio codec
// are deliberately separate from the non-causal Qwen3 backbone implemented here.
package omnivoice

import (
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	config "github.com/rcarmo/go-pherence/loader/omnivoice"
)

var blockPrepackedSuffixes = [...]string{
	"self_attn.q_proj.weight",
	"self_attn.k_proj.weight",
	"self_attn.v_proj.weight",
	"self_attn.o_proj.weight",
	"mlp.gate_proj.weight",
	"mlp.up_proj.weight",
	"mlp.down_proj.weight",
}

func blockWeightShapes(c config.LLMConfig) map[string][2]int {
	return map[string][2]int{
		"input_layernorm.weight":          {1, c.HiddenSize},
		"post_attention_layernorm.weight": {1, c.HiddenSize},
		"self_attn.q_norm.weight":         {1, c.HeadDim},
		"self_attn.k_norm.weight":         {1, c.HeadDim},
		"self_attn.q_proj.weight":         {c.NumAttentionHeads * c.HeadDim, c.HiddenSize},
		"self_attn.k_proj.weight":         {c.NumKeyValueHeads * c.HeadDim, c.HiddenSize},
		"self_attn.v_proj.weight":         {c.NumKeyValueHeads * c.HeadDim, c.HiddenSize},
		"self_attn.o_proj.weight":         {c.HiddenSize, c.NumAttentionHeads * c.HeadDim},
		"mlp.gate_proj.weight":            {c.IntermediateSize, c.HiddenSize},
		"mlp.up_proj.weight":              {c.IntermediateSize, c.HiddenSize},
		"mlp.down_proj.weight":            {c.HiddenSize, c.IntermediateSize},
	}
}

// Block contains one Qwen3 decoder layer. Weights are row-major [out,in].
// Construct through NewBlock; weight slices must not be modified during use.
type Block struct {
	config    config.LLMConfig
	weights   map[string][]float32
	prepacked map[string][]float32
	directQ8  map[string][]byte
}

// ValidateConfig checks native block support without loading any weights.
func ValidateConfig(c config.LLMConfig) error {
	if err := c.Validate(); err != nil {
		return err
	}
	if c.NumAttentionHeads%c.NumKeyValueHeads != 0 || c.HeadDim%2 != 0 {
		return fmt.Errorf("omnivoice: unsupported GQA/rotary dimensions")
	}
	if c.AttentionBias || c.HiddenAct != "silu" || c.RopeParameters.RopeType != "default" || c.UseSlidingWindow {
		return fmt.Errorf("omnivoice: only bias-free SiLU, default RoPE, full attention supported")
	}
	if c.RMSNormEps <= 0 || math.IsNaN(c.RMSNormEps) || math.IsInf(c.RMSNormEps, 0) {
		return fmt.Errorf("omnivoice: invalid normalization epsilon")
	}
	return nil
}

// NewBlock validates all dimensions before permitting SIMD operations.
func NewBlock(c config.LLMConfig, weights map[string][]float32) (*Block, error) {
	if err := ValidateConfig(c); err != nil {
		return nil, err
	}
	sizes := blockWeightShapes(c)
	copyWeights := make(map[string][]float32, len(sizes))
	for name, shape := range sizes {
		n, ok := product(shape[0], shape[1])
		if !ok || len(weights[name]) != n {
			return nil, fmt.Errorf("omnivoice: %s has %d elements, expected %v", name, len(weights[name]), shape)
		}
		copyWeights[name] = weights[name]
	}
	return &Block{config: c, weights: copyWeights}, nil
}

func (b *Block) cloneWithPrepacked(prepacked map[string][]float32) *Block {
	if b == nil {
		return nil
	}
	clone := *b
	clone.prepacked = prepacked
	return &clone
}

func product(a, b int) (int, bool) {
	if a <= 0 || b <= 0 || a > int(^uint(0)>>1)/b {
		return 0, false
	}
	return a * b, true
}

// Forward computes a batch-one, full-attention decoder block. x is [tokens,hidden].
// positions may be nil (0..tokens-1). mask is nil or a row-major [tokens,tokens]
// additive attention bias; use -Inf to prohibit an edge. Every query must have
// at least one finite edge. Unlike autoregressive LLMs, no causal mask is added.
// Input and output never alias. No KV cache is retained across denoising steps.
func (b *Block) Forward(x []float32, tokens int, positions []int, mask []float32) ([]float32, error) {
	if err := b.validateInput(x, tokens, positions, mask); err != nil {
		return nil, err
	}
	scratch, err := b.NewWorkspace(tokens)
	if err != nil {
		return nil, err
	}
	dst := make([]float32, len(x))
	if err = b.ForwardInto(dst, x, tokens, positions, mask, scratch); err != nil {
		return nil, err
	}
	return dst, nil
}

func (b *Block) validateInput(x []float32, tokens int, positions []int, mask []float32) error {
	c := b.config
	h := c.HiddenSize
	d := c.HeadDim
	nh := c.NumAttentionHeads
	nkv := c.NumKeyValueHeads
	n, ok := product(tokens, h)
	if !ok || len(x) != n {
		return fmt.Errorf("omnivoice: invalid input shape")
	}
	square, ok := product(tokens, tokens)
	if !ok {
		return fmt.Errorf("omnivoice: sequence size overflow")
	}
	if positions != nil && len(positions) != tokens {
		return fmt.Errorf("omnivoice: position length mismatch")
	}
	for _, p := range positions {
		if p < 0 {
			return fmt.Errorf("omnivoice: negative position")
		}
	}
	if mask != nil {
		if len(mask) != square {
			return fmt.Errorf("omnivoice: mask shape mismatch")
		}
		for i := 0; i < tokens; i++ {
			finite := false
			for _, v := range mask[i*tokens : (i+1)*tokens] {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 1) {
					return fmt.Errorf("omnivoice: invalid mask")
				}
				finite = finite || !math.IsInf(float64(v), -1)
			}
			if !finite {
				return fmt.Errorf("omnivoice: entirely masked query")
			}
		}
	}
	for _, dim := range []int{nh * d, nkv * d, c.IntermediateSize} {
		if _, ok := product(tokens, dim); !ok {
			return fmt.Errorf("omnivoice: intermediate shape overflow")
		}
	}
	return nil
}

func (b *Block) linearInto(s *Workspace, y, x, w []float32, key string, rows, in, out int) error {
	if raw := b.directQ8[key]; raw != nil {
		if !simd.SgemmNTQ8_0To(y, x, raw, rows, out, in) {
			return fmt.Errorf("omnivoice: invalid direct Q8 projection")
		}
		return nil
	}
	if s != nil && s.pool != nil && s.executionContext != nil {
		clear(y)
		return s.poolLinearInto(y, x, w, b.prepacked[key], rows, in, out)
	}
	if packed := b.prepacked[key]; len(packed) != 0 {
		clear(y)
		if !simd.SgemmNTPrepackedTo(y, x, w, packed, rows, out, in, 1, in, in, out) {
			panic("omnivoice: internal linear shape error")
		}
		return nil
	}
	s.linearInto(y, x, w, rows, in, out)
	return nil
}

// ForwardInto is the allocation-free block path after workspace construction.
// Scratch is exclusive to this call; dst may equal x exactly, but must not
// otherwise overlap inputs, weights, or workspace. Errors leave dst unspecified.
func (b *Block) ForwardInto(dst, x []float32, tokens int, positions []int, mask []float32, s *Workspace) error {
	if err := b.validateInput(x, tokens, positions, mask); err != nil {
		return err
	}
	if !s.matches(b, tokens) || len(dst) != len(x) {
		return fmt.Errorf("omnivoice: workspace/output shape mismatch")
	}
	c := b.config
	h, d, nh, nkv := c.HiddenSize, c.HeadDim, c.NumAttentionHeads, c.NumKeyValueHeads
	w := b.weights
	norm, q, k, v, attended := s.norm, s.q, s.k, s.v, s.attended
	normalizeInto(norm, x, w["input_layernorm.weight"], tokens, h, float32(c.RMSNormEps))
	if err := b.linearInto(s, q, norm, w["self_attn.q_proj.weight"], "self_attn.q_proj.weight", tokens, h, nh*d); err != nil {
		return err
	}
	if err := b.linearInto(s, k, norm, w["self_attn.k_proj.weight"], "self_attn.k_proj.weight", tokens, h, nkv*d); err != nil {
		return err
	}
	if err := b.linearInto(s, v, norm, w["self_attn.v_proj.weight"], "self_attn.v_proj.weight", tokens, h, nkv*d); err != nil {
		return err
	}
	normalizeInto(q, q, w["self_attn.q_norm.weight"], tokens*nh, d, float32(c.RMSNormEps))
	normalizeInto(k, k, w["self_attn.k_norm.weight"], tokens*nkv, d, float32(c.RMSNormEps))
	s.prepareRoPE(positions)
	s.rotate(q, nh)
	s.rotate(k, nkv)
	// Pack each head contiguously so QK^T and PV use existing SIMD GEMM.
	// Only one tokens^2 score buffer is live, not heads*tokens^2.
	qhead, khead, vhead := s.qhead, s.khead, s.vhead
	scores, headout := s.scores, s.headout
	for head := 0; head < nh; head++ {
		kh := head / (nh / nkv)
		for t := 0; t < tokens; t++ {
			copy(qhead[t*d:(t+1)*d], q[(t*nh+head)*d:(t*nh+head+1)*d])
			copy(khead[t*d:(t+1)*d], k[(t*nkv+kh)*d:(t*nkv+kh+1)*d])
			copy(vhead[t*d:(t+1)*d], v[(t*nkv+kh)*d:(t*nkv+kh+1)*d])
		}
		clear(scores)
		simd.SgemmNTTo(scores, qhead, khead, tokens, tokens, d, float32(1/math.Sqrt(float64(d))), d, d, tokens)
		if mask != nil {
			simd.VecAdd(scores, scores, mask)
		}
		for t := 0; t < tokens; t++ {
			if !simd.SoftmaxSIMDInPlace(scores[t*tokens : (t+1)*tokens]) {
				return fmt.Errorf("omnivoice: attention softmax failed")
			}
		}
		clear(headout)
		simd.SgemmNNTo(headout, scores, vhead, tokens, d, tokens, 1, tokens, d, d)
		for t := 0; t < tokens; t++ {
			copy(attended[(t*nh+head)*d:(t*nh+head+1)*d], headout[t*d:(t+1)*d])
		}
	}
	if err := b.linearInto(s, s.down, attended, w["self_attn.o_proj.weight"], "self_attn.o_proj.weight", tokens, nh*d, h); err != nil {
		return err
	}
	simd.VecAdd(dst, s.down, x)
	normalizeInto(norm, dst, w["post_attention_layernorm.weight"], tokens, h, float32(c.RMSNormEps))
	if err := b.linearInto(s, s.gate, norm, w["mlp.gate_proj.weight"], "mlp.gate_proj.weight", tokens, h, c.IntermediateSize); err != nil {
		return err
	}
	if err := b.linearInto(s, s.up, norm, w["mlp.up_proj.weight"], "mlp.up_proj.weight", tokens, h, c.IntermediateSize); err != nil {
		return err
	}
	// GEMM packing scratch is idle during activation; process bounded tiles
	// rather than reserving another full feed-forward activation tensor.
	for start := 0; start < len(s.gate); start += len(s.packed) {
		end := min(start+len(s.packed), len(s.gate))
		if !simd.SiLUMulExpTo(s.gate[start:end], s.gate[start:end], s.up[start:end], s.packed[:end-start]) {
			return fmt.Errorf("omnivoice: SiLU scratch shape failed")
		}
	}
	if err := b.linearInto(s, s.down, s.gate, w["mlp.down_proj.weight"], "mlp.down_proj.weight", tokens, c.IntermediateSize, h); err != nil {
		return err
	}
	simd.VecAdd(dst, dst, s.down)
	return nil
}
