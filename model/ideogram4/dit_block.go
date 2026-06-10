package ideogram4

import (
	"fmt"
	"math"
)

// DiTLayer bundles the FP8 linears and RMSNorm weights of one Ideogram4
// transformer block, matching the reference Ideogram4TransformerBlock.
type DiTLayer struct {
	Index int

	QKV   *FP8Linear // emb -> 3*emb (bias=false)
	O     *FP8Linear // emb -> emb   (bias=false)
	W1    *FP8Linear // emb -> intermediate
	W2    *FP8Linear // intermediate -> emb
	W3    *FP8Linear // emb -> intermediate
	AdaLN *FP8Linear // adaln_dim -> 4*emb (bias=true)

	NormQ  []float32 // head_dim RMSNorm (eps 1e-5)
	NormK  []float32 // head_dim RMSNorm (eps 1e-5)
	AttnN1 []float32 // emb RMSNorm (norm_eps)
	AttnN2 []float32 // emb RMSNorm (norm_eps)
	FfnN1  []float32 // emb RMSNorm (norm_eps)
	FfnN2  []float32 // emb RMSNorm (norm_eps)

	gpu *ditLayerGPUResidency // optional cached FP8 residency for windowed layers
}

// MRoPE holds precomputed cos/sin tables for the Ideogram4 3-section
// interleaved rotary embedding. The rotary covers the full head_dim via
// head_dim/2 frequency pairs (NeoX rotate-half), where each pair's angle is
// driven by the temporal, height, or width coordinate per the interleaved
// mrope_section assignment.
type MRoPE struct {
	headDim int
	half    int       // head_dim/2
	cos     []float32 // [tokens, half]
	sin     []float32 // [tokens, half]
	tokens  int
}

// BuildMRoPE precomputes rotary tables for image tokens over a gridH x gridW
// latent grid (single temporal frame).
func BuildMRoPE(cfg Config, gridH, gridW int) (*MRoPE, error) {
	if gridH <= 0 || gridW <= 0 {
		return nil, fmt.Errorf("ideogram4 mrope grid %dx%d", gridH, gridW)
	}
	positions := make([][3]int, 0, gridH*gridW)
	for r := 0; r < gridH; r++ {
		for c := 0; c < gridW; c++ {
			positions = append(positions, [3]int{0, r, c})
		}
	}
	return BuildMRoPEPositions(cfg, positions)
}

// BuildMRoPEPositions precomputes rotary tables for explicit per-token
// (temporal, height, width) coordinates, matching the reference Ideogram4MRoPE:
// inv_freq[j] = base^(-2j/head_dim); the interleaved section assignment routes
// j%3==1 (j<3*section_h) to the height axis and j%3==2 (j<3*section_w) to the
// width axis; all other freqs use the temporal axis.
func BuildMRoPEPositions(cfg Config, positions [][3]int) (*MRoPE, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if len(cfg.MRoPESection) != 3 {
		return nil, fmt.Errorf("ideogram4 mrope section=%v want 3 entries", cfg.MRoPESection)
	}
	if cfg.HeadDim%2 != 0 {
		return nil, fmt.Errorf("ideogram4 mrope head_dim=%d not even", cfg.HeadDim)
	}
	if len(positions) == 0 {
		return nil, fmt.Errorf("ideogram4 mrope empty positions")
	}
	theta := float64(cfg.RopeTheta)
	if theta <= 0 {
		theta = 5000000
	}
	half := cfg.HeadDim / 2
	// axis selector per freq index j.
	axisOf := make([]int, half)
	hLen := cfg.MRoPESection[1] * 3
	wLen := cfg.MRoPESection[2] * 3
	for j := 0; j < half; j++ {
		switch {
		case j%3 == 1 && j < hLen:
			axisOf[j] = 1
		case j%3 == 2 && j < wLen:
			axisOf[j] = 2
		default:
			axisOf[j] = 0
		}
	}
	invFreq := make([]float64, half)
	for j := 0; j < half; j++ {
		invFreq[j] = math.Pow(theta, -float64(2*j)/float64(cfg.HeadDim))
	}
	tokens := len(positions)
	rope := &MRoPE{headDim: cfg.HeadDim, half: half, tokens: tokens,
		cos: make([]float32, tokens*half), sin: make([]float32, tokens*half)}
	for tok, p := range positions {
		for j := 0; j < half; j++ {
			pos := float64(p[axisOf[j]])
			ang := pos * invFreq[j]
			rope.cos[tok*half+j] = float32(math.Cos(ang))
			rope.sin[tok*half+j] = float32(math.Sin(ang))
		}
	}
	return rope, nil
}

// applyToHead rotates one head vector in place using NeoX rotate-half over the
// full head_dim: pair (j, j+half) rotates by angle j.
func (m *MRoPE) applyToHead(vec []float32, token int) {
	base := token * m.half
	for j := 0; j < m.half; j++ {
		x1 := vec[j]
		x2 := vec[m.half+j]
		c := m.cos[base+j]
		s := m.sin[base+j]
		vec[j] = x1*c - x2*s
		vec[m.half+j] = x2*c + x1*s
	}
}

func applyMRoPEToQK(q, k []float32, rope *MRoPE, tokens, heads, headDim int) error {
	if gpuMRoPEEnabled() {
		qGPU := append([]float32(nil), q...)
		kGPU := append([]float32(nil), k...)
		qErr := applyMRoPEGPU(qGPU, rope, tokens, heads, headDim)
		kErr := applyMRoPEGPU(kGPU, rope, tokens, heads, headDim)
		if qErr == nil && kErr == nil {
			copy(q, qGPU)
			copy(k, kGPU)
			return nil
		}
		if gpuMRoPEStrict() {
			return fmt.Errorf("ideogram4 GPU MRoPE q=%v k=%v", qErr, kErr)
		}
	}
	for t := 0; t < tokens; t++ {
		for h := 0; h < heads; h++ {
			off := t*heads*headDim + h*headDim
			rope.applyToHead(q[off:off+headDim], t)
			rope.applyToHead(k[off:off+headDim], t)
		}
	}
	return nil
}

// ForwardLayer applies one DiT block to hidden states in place. hidden is
// [tokens, emb] row-major. adalnInput is SiLU(adaln_proj(t_embedding)) of length
// AdaLNDim. rope must be built for the same token set.
//
// Reference Ideogram4TransformerBlock:
//
//	mod = adaln_modulation(adaln_input); scale_msa, gate_msa, scale_mlp, gate_mlp = chunk(4)
//	gate = tanh(gate); scale = 1 + scale
//	attn = attention(attention_norm1(x) * scale_msa)
//	x = x + gate_msa * attention_norm2(attn)
//	x = x + gate_mlp * ffn_norm2(feed_forward(ffn_norm1(x) * scale_mlp))
func (l *DiTLayer) ForwardLayer(cfg Config, hidden []float32, adalnInput []float32, rope *MRoPE) error {
	if err := cfg.Validate(); err != nil {
		return err
	}
	emb := cfg.EmbDim
	if len(hidden)%emb != 0 {
		return fmt.Errorf("ideogram4 DiT hidden len=%d not divisible by emb=%d", len(hidden), emb)
	}
	tokens := len(hidden) / emb
	if rope == nil || rope.tokens != tokens {
		return fmt.Errorf("ideogram4 DiT rope tokens mismatch")
	}
	if l.AdaLN.OutDim() != 4*emb {
		return fmt.Errorf("ideogram4 DiT adaln out=%d want=%d", l.AdaLN.OutDim(), 4*emb)
	}
	layerGPU, err := l.uploadGPU()
	if err != nil {
		if gpuFP8Strict() {
			return err
		}
		layerGPU = nil
	}
	if !l.cacheGPUResidency() {
		defer layerGPU.Free()
	}
	mod := make([]float32, 4*emb)
	if err := layerGPU.AdaLN(*l, adalnInput, mod); err != nil {
		return err
	}
	scaleMSA := mod[0:emb]
	gateMSA := mod[emb : 2*emb]
	scaleMLP := mod[2*emb : 3*emb]
	gateMLP := mod[3*emb : 4*emb]
	transformAdaLNMod(mod, emb)

	normEps := float32(cfg.NormEps)
	if normEps <= 0 {
		normEps = 1e-5
	}
	heads, headDim := cfg.NumHeads, cfg.HeadDim
	scaleAttn := float32(1 / math.Sqrt(float64(headDim)))

	// ---- Attention sublayer ----
	normedAll := make([]float32, tokens*emb)
	if gpuNormEnabled() {
		if err := rmsNormRowsWeightedGPU(normedAll, hidden, l.AttnN1, scaleMSA, tokens, emb, normEps); err != nil && gpuNormStrict() {
			return err
		} else if err != nil {
			for t := 0; t < tokens; t++ {
				row := hidden[t*emb : (t+1)*emb]
				normed := normedAll[t*emb : (t+1)*emb]
				rmsNormWeightedCPU(normed, row, l.AttnN1, normEps)
				for i := 0; i < emb; i++ {
					normed[i] *= scaleMSA[i]
				}
			}
		}
	} else {
		for t := 0; t < tokens; t++ {
			row := hidden[t*emb : (t+1)*emb]
			normed := normedAll[t*emb : (t+1)*emb]
			rmsNormWeightedCPU(normed, row, l.AttnN1, normEps)
			for i := 0; i < emb; i++ {
				normed[i] *= scaleMSA[i]
			}
		}
	}
	// QKV projection through attention, O projection, post-norm, and gated
	// residual. Keep this full attention sublayer on-device when possible and
	// download updated hidden once.
	if err := layerGPU.AttentionResidualBatch(*l, hidden, normedAll, gateMSA, tokens, heads, headDim, rope, scaleAttn, normEps); err != nil {
		return err
	}

	// ---- MLP sublayer (SwiGLU) ----
	inter := cfg.IntermediateSize
	mlpIn := normedAll
	if gpuNormEnabled() {
		if err := rmsNormRowsWeightedGPU(mlpIn, hidden, l.FfnN1, scaleMLP, tokens, emb, normEps); err != nil && gpuNormStrict() {
			return err
		} else if err != nil {
			for t := 0; t < tokens; t++ {
				row := hidden[t*emb : (t+1)*emb]
				normed := mlpIn[t*emb : (t+1)*emb]
				rmsNormWeightedCPU(normed, row, l.FfnN1, normEps)
				for i := 0; i < emb; i++ {
					normed[i] *= scaleMLP[i]
				}
			}
		}
	} else {
		for t := 0; t < tokens; t++ {
			row := hidden[t*emb : (t+1)*emb]
			normed := mlpIn[t*emb : (t+1)*emb]
			rmsNormWeightedCPU(normed, row, l.FfnN1, normEps)
			for i := 0; i < emb; i++ {
				normed[i] *= scaleMLP[i]
			}
		}
	}
	_ = inter
	if gpuFP8Enabled() && gpuNormEnabled() {
		if err := layerGPU.MLPResidualBatch(*l, hidden, mlpIn, gateMLP, tokens); err != nil && (gpuFP8Strict() || gpuNormStrict() || gpuMLPStrict()) {
			return err
		} else if err == nil {
			return nil
		}
	}
	downAll := make([]float32, tokens*emb)
	postNormAll := make([]float32, tokens*emb)
	if err := layerGPU.MLPBatch(*l, mlpIn, downAll, tokens); err != nil {
		return err
	}
	if gpuNormEnabled() {
		if err := rmsNormRowsWeightedGPU(postNormAll, downAll, l.FfnN2, nil, tokens, emb, normEps); err != nil && gpuNormStrict() {
			return err
		} else if err != nil {
			for t := 0; t < tokens; t++ {
				rmsNormWeightedCPU(postNormAll[t*emb:(t+1)*emb], downAll[t*emb:(t+1)*emb], l.FfnN2, normEps)
			}
		}
		if err := gatedResidualRowsGPU(hidden, postNormAll, gateMLP, tokens, emb); err != nil && gpuNormStrict() {
			return err
		} else if err != nil {
			for t := 0; t < tokens; t++ {
				row := hidden[t*emb : (t+1)*emb]
				upd := postNormAll[t*emb : (t+1)*emb]
				for i := range row {
					row[i] += gateMLP[i] * upd[i]
				}
			}
		}
	} else {
		for t := 0; t < tokens; t++ {
			rmsNormWeightedCPU(postNormAll[t*emb:(t+1)*emb], downAll[t*emb:(t+1)*emb], l.FfnN2, normEps)
			row := hidden[t*emb : (t+1)*emb]
			upd := postNormAll[t*emb : (t+1)*emb]
			for i := range row {
				row[i] += gateMLP[i] * upd[i]
			}
		}
	}
	return nil
}

func siluMulInPlace(gate, up []float32) {
	if gpuMLPEnabled() {
		out := make([]float32, len(gate))
		if err := siluMulGPU(out, gate, up); err == nil {
			copy(gate, out)
			return
		} else if gpuMLPStrict() {
			panic(fmt.Sprintf("ideogram4 GPU SiLU*Mul: %v", err))
		}
	}
	for i := range gate {
		gate[i] = siluScalar(gate[i]) * up[i]
	}
}

func fullSelfAttention(attnOut, q, k, v []float32, tokens, heads, headDim int, scaleAttn float32) error {
	if gpuAttentionEnabled() {
		if err := fullAttentionGPU(attnOut, q, k, v, tokens, heads, headDim, scaleAttn); err == nil || gpuAttentionStrict() {
			return err
		}
	}
	scores := make([]float32, tokens)
	emb := heads * headDim
	for h := 0; h < heads; h++ {
		for ti := 0; ti < tokens; ti++ {
			qoff := ti*emb + h*headDim
			for tj := 0; tj < tokens; tj++ {
				koff := tj*emb + h*headDim
				var dot float32
				for d := 0; d < headDim; d++ {
					dot += q[qoff+d] * k[koff+d]
				}
				scores[tj] = dot * scaleAttn
			}
			softmaxFallback(scores)
			off := ti*emb + h*headDim
			for tj := 0; tj < tokens; tj++ {
				w := scores[tj]
				voff := tj*emb + h*headDim
				for d := 0; d < headDim; d++ {
					attnOut[off+d] += w * v[voff+d]
				}
			}
		}
	}
	return nil
}

func transformAdaLNMod(mod []float32, emb int) {
	if gpuNormEnabled() {
		if err := adalnTransformGPU(mod, emb); err == nil || gpuNormStrict() {
			return
		}
	}
	scaleMSA := mod[0:emb]
	gateMSA := mod[emb : 2*emb]
	scaleMLP := mod[2*emb : 3*emb]
	gateMLP := mod[3*emb : 4*emb]
	for i := 0; i < emb; i++ {
		scaleMSA[i] = 1 + scaleMSA[i]
		scaleMLP[i] = 1 + scaleMLP[i]
		gateMSA[i] = tanh32(gateMSA[i])
		gateMLP[i] = tanh32(gateMLP[i])
	}
}

func addGatedResidual(row, update, gate []float32) {
	if gpuNormEnabled() {
		if err := gatedResidualGPU(row, update, gate); err == nil || gpuNormStrict() {
			return
		}
	}
	for i := range row {
		row[i] += gate[i] * update[i]
	}
}

// rmsNormWeightedTo computes RMSNorm(x)*weight into dst.
func rmsNormWeightedTo(dst, x, weight []float32, eps float32) {
	if gpuNormEnabled() {
		if err := rmsNormWeightedGPU(dst, x, weight, eps); err == nil || gpuNormStrict() {
			return
		}
	}
	rmsNormWeightedCPU(dst, x, weight, eps)
}

func rmsNormWeightedInPlace(x, weight []float32, eps float32) {
	if gpuNormEnabled() {
		dst := make([]float32, len(x))
		if err := rmsNormWeightedGPU(dst, x, weight, eps); err == nil || gpuNormStrict() {
			copy(x, dst)
			return
		}
	}
	rmsNormWeightedCPU(x, x, weight, eps)
}

func rmsNormWeightedCPU(dst, x, weight []float32, eps float32) {
	var ss float64
	for _, vv := range x {
		ss += float64(vv) * float64(vv)
	}
	inv := float32(1 / math.Sqrt(ss/float64(len(x))+float64(eps)))
	for i := range x {
		dst[i] = x[i] * inv * weight[i]
	}
}

func tanh32(x float32) float32 { return float32(math.Tanh(float64(x))) }

func siluScalar(x float32) float32 {
	return x / (1 + float32(math.Exp(float64(-x))))
}

func softmaxFallback(x []float32) {
	max := x[0]
	for _, v := range x {
		if v > max {
			max = v
		}
	}
	var sum float32
	for i := range x {
		x[i] = float32(math.Exp(float64(x[i] - max)))
		sum += x[i]
	}
	for i := range x {
		x[i] /= sum
	}
}
