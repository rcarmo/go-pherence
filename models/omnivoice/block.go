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

// Block contains one Qwen3 decoder layer. Weights are row-major [out,in].
// Construct through NewBlock; weight slices must not be modified during use.
type Block struct {
	config  config.LLMConfig
	weights map[string][]float32
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
	sizes := map[string][2]int{
		"input_layernorm.weight": {1, c.HiddenSize}, "post_attention_layernorm.weight": {1, c.HiddenSize},
		"self_attn.q_norm.weight": {1, c.HeadDim}, "self_attn.k_norm.weight": {1, c.HeadDim},
		"self_attn.q_proj.weight": {c.NumAttentionHeads * c.HeadDim, c.HiddenSize},
		"self_attn.k_proj.weight": {c.NumKeyValueHeads * c.HeadDim, c.HiddenSize},
		"self_attn.v_proj.weight": {c.NumKeyValueHeads * c.HeadDim, c.HiddenSize},
		"self_attn.o_proj.weight": {c.HiddenSize, c.NumAttentionHeads * c.HeadDim},
		"mlp.gate_proj.weight":    {c.IntermediateSize, c.HiddenSize}, "mlp.up_proj.weight": {c.IntermediateSize, c.HiddenSize},
		"mlp.down_proj.weight": {c.HiddenSize, c.IntermediateSize},
	}
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
	c := b.config
	h := c.HiddenSize
	d := c.HeadDim
	nh := c.NumAttentionHeads
	nkv := c.NumKeyValueHeads
	n, ok := product(tokens, h)
	if !ok || len(x) != n {
		return nil, fmt.Errorf("omnivoice: invalid input shape")
	}
	square, ok := product(tokens, tokens)
	if !ok {
		return nil, fmt.Errorf("omnivoice: sequence size overflow")
	}
	if positions != nil && len(positions) != tokens {
		return nil, fmt.Errorf("omnivoice: position length mismatch")
	}
	for _, p := range positions {
		if p < 0 {
			return nil, fmt.Errorf("omnivoice: negative position")
		}
	}
	if mask != nil {
		if len(mask) != square {
			return nil, fmt.Errorf("omnivoice: mask shape mismatch")
		}
		for i := 0; i < tokens; i++ {
			finite := false
			for _, v := range mask[i*tokens : (i+1)*tokens] {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 1) {
					return nil, fmt.Errorf("omnivoice: invalid mask")
				}
				finite = finite || !math.IsInf(float64(v), -1)
			}
			if !finite {
				return nil, fmt.Errorf("omnivoice: entirely masked query")
			}
		}
	}
	for _, dim := range []int{nh * d, nkv * d, c.IntermediateSize} {
		if _, ok := product(tokens, dim); !ok {
			return nil, fmt.Errorf("omnivoice: intermediate shape overflow")
		}
	}
	w := b.weights
	norm := normalize(x, w["input_layernorm.weight"], tokens, h, float32(c.RMSNormEps))
	q := linear(norm, w["self_attn.q_proj.weight"], tokens, h, nh*d)
	k := linear(norm, w["self_attn.k_proj.weight"], tokens, h, nkv*d)
	v := linear(norm, w["self_attn.v_proj.weight"], tokens, h, nkv*d)
	q = normalize(q, w["self_attn.q_norm.weight"], tokens*nh, d, float32(c.RMSNormEps))
	k = normalize(k, w["self_attn.k_norm.weight"], tokens*nkv, d, float32(c.RMSNormEps))
	rope(q, tokens, nh, d, positions, c.RopeParameters.RopeTheta)
	rope(k, tokens, nkv, d, positions, c.RopeParameters.RopeTheta)
	attended := make([]float32, tokens*nh*d)
	// Pack each head contiguously so QK^T and PV use existing SIMD GEMM.
	// Only one tokens^2 score buffer is live, not heads*tokens^2.
	qhead, khead, vhead := make([]float32, tokens*d), make([]float32, tokens*d), make([]float32, tokens*d)
	scores, headout := make([]float32, square), make([]float32, tokens*d)
	for head := 0; head < nh; head++ {
		kh := head / (nh / nkv)
		for t := 0; t < tokens; t++ {
			copy(qhead[t*d:(t+1)*d], q[(t*nh+head)*d:(t*nh+head+1)*d])
			copy(khead[t*d:(t+1)*d], k[(t*nkv+kh)*d:(t*nkv+kh+1)*d])
			copy(vhead[t*d:(t+1)*d], v[(t*nkv+kh)*d:(t*nkv+kh+1)*d])
		}
		clear(scores)
		simd.SgemmNTTo(scores, qhead, khead, tokens, tokens, d, float32(1/math.Sqrt(float64(d))), d, d, tokens)
		for i := range scores {
			if mask != nil {
				scores[i] += mask[i]
			}
		}
		for t := 0; t < tokens; t++ {
			if !simd.SoftmaxInPlace(scores[t*tokens : (t+1)*tokens]) {
				return nil, fmt.Errorf("omnivoice: attention softmax failed")
			}
		}
		clear(headout)
		simd.SgemmNNTo(headout, scores, vhead, tokens, d, tokens, 1, tokens, d, d)
		for t := 0; t < tokens; t++ {
			copy(attended[(t*nh+head)*d:(t*nh+head+1)*d], headout[t*d:(t+1)*d])
		}
	}
	out := linear(attended, w["self_attn.o_proj.weight"], tokens, nh*d, h)
	for i := range out {
		out[i] += x[i]
	}
	norm = normalize(out, w["post_attention_layernorm.weight"], tokens, h, float32(c.RMSNormEps))
	gate := linear(norm, w["mlp.gate_proj.weight"], tokens, h, c.IntermediateSize)
	up := linear(norm, w["mlp.up_proj.weight"], tokens, h, c.IntermediateSize)
	simd.SiLUMul(gate, gate, up)
	down := linear(gate, w["mlp.down_proj.weight"], tokens, c.IntermediateSize, h)
	for i := range out {
		out[i] += down[i]
	}
	return out, nil
}

func linear(x, w []float32, rows, in, out int) []float32 {
	y := make([]float32, rows*out)
	if !simd.SgemmNTTo(y, x, w, rows, out, in, 1, in, in, out) {
		panic("omnivoice: internal linear shape error")
	}
	return y
}
func normalize(x, w []float32, rows, width int, eps float32) []float32 {
	y := make([]float32, len(x))
	for r := 0; r < rows; r++ {
		copy(y[r*width:(r+1)*width], x[r*width:(r+1)*width])
		simd.RMSNorm(y[r*width:(r+1)*width], w, eps)
	}
	return y
}
func rope(x []float32, tokens, heads, dim int, positions []int, theta float64) {
	for t := 0; t < tokens; t++ {
		p := t
		if positions != nil {
			p = positions[t]
		}
		for j := 0; j < dim/2; j++ {
			angle := float32(p) * float32(1/math.Pow(theta, float64(2*j)/float64(dim)))
			co, si := float32(math.Cos(float64(angle))), float32(math.Sin(float64(angle)))
			for head := 0; head < heads; head++ {
				a := (t*heads+head)*dim + j
				b := a + dim/2
				u, v := x[a], x[b]
				x[a] = u*co - v*si
				x[b] = v*co + u*si
			}
		}
	}
}
