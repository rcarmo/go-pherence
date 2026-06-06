package ideogram4

import (
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

// DiTLayer bundles the six FP8 linear matrices of one Ideogram4 transformer
// block, as enumerated by RequiredLinearSpecs.
type DiTLayer struct {
	AdaLN *FP8Linear // adaln_dim -> 4*emb (shift/scale for attn + mlp)
	QKV   *FP8Linear // emb -> 3*emb
	O     *FP8Linear // emb -> emb
	W1    *FP8Linear // emb -> intermediate (gate)
	W2    *FP8Linear // intermediate -> emb (down)
	W3    *FP8Linear // emb -> intermediate (up)
}

// DiTLayerFromSet pulls the six linears for a layer index out of a loaded set.
func DiTLayerFromSet(set map[string]*FP8Linear, layer int) (DiTLayer, error) {
	get := func(suffix string) (*FP8Linear, error) {
		key := fmt.Sprintf("layers.%d.%s", layer, suffix)
		lin, ok := set[key]
		if !ok || lin == nil {
			return nil, fmt.Errorf("ideogram4 DiT layer %d missing %q", layer, key)
		}
		return lin, nil
	}
	var l DiTLayer
	var err error
	if l.AdaLN, err = get("adaln_modulation"); err != nil {
		return DiTLayer{}, err
	}
	if l.QKV, err = get("attention.qkv"); err != nil {
		return DiTLayer{}, err
	}
	if l.O, err = get("attention.o"); err != nil {
		return DiTLayer{}, err
	}
	if l.W1, err = get("feed_forward.w1"); err != nil {
		return DiTLayer{}, err
	}
	if l.W2, err = get("feed_forward.w2"); err != nil {
		return DiTLayer{}, err
	}
	if l.W3, err = get("feed_forward.w3"); err != nil {
		return DiTLayer{}, err
	}
	return l, nil
}

// MRoPE holds precomputed cos/sin tables for the Ideogram4 3-section rotary
// embedding (temporal/height/width). The rotary covers 2*sum(section) head
// dims; the remainder of each head is left unrotated.
type MRoPE struct {
	headDim  int
	rotPairs int       // sum(section)
	cos      []float32 // [tokens, rotPairs]
	sin      []float32 // [tokens, rotPairs]
	tokens   int
}

// BuildMRoPE precomputes rotary tables for image tokens laid out row-major over
// a gridH x gridW latent grid (single temporal frame). Section assigns the
// first s0 freq pairs to the temporal axis, the next s1 to height, the last s2
// to width.
func BuildMRoPE(cfg Config, gridH, gridW int) (*MRoPE, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if len(cfg.MRoPESection) != 3 {
		return nil, fmt.Errorf("ideogram4 mrope section=%v want 3 entries", cfg.MRoPESection)
	}
	rotPairs := 0
	for _, s := range cfg.MRoPESection {
		if s < 0 {
			return nil, fmt.Errorf("ideogram4 mrope section negative %v", cfg.MRoPESection)
		}
		rotPairs += s
	}
	if 2*rotPairs > cfg.HeadDim {
		return nil, fmt.Errorf("ideogram4 mrope rot dims=%d exceed head_dim=%d", 2*rotPairs, cfg.HeadDim)
	}
	theta := float64(cfg.RopeTheta)
	if theta <= 0 {
		theta = 5000000
	}
	tokens := gridH * gridW
	rope := &MRoPE{headDim: cfg.HeadDim, rotPairs: rotPairs, tokens: tokens,
		cos: make([]float32, tokens*rotPairs), sin: make([]float32, tokens*rotPairs)}
	// inverse frequencies over the full rotary span (2*rotPairs dims).
	invFreq := make([]float64, rotPairs)
	for i := 0; i < rotPairs; i++ {
		invFreq[i] = math.Pow(theta, -float64(2*i)/float64(2*rotPairs))
	}
	s0, s1 := cfg.MRoPESection[0], cfg.MRoPESection[1]
	for r := 0; r < gridH; r++ {
		for c := 0; c < gridW; c++ {
			tok := r*gridW + c
			for i := 0; i < rotPairs; i++ {
				var pos float64 // axis coordinate per section
				switch {
				case i < s0:
					pos = 0 // single temporal frame
				case i < s0+s1:
					pos = float64(r)
				default:
					pos = float64(c)
				}
				ang := pos * invFreq[i]
				rope.cos[tok*rotPairs+i] = float32(math.Cos(ang))
				rope.sin[tok*rotPairs+i] = float32(math.Sin(ang))
			}
		}
	}
	return rope, nil
}

// applyToHead rotates the first 2*rotPairs dims of one head vector in place
// using the rotate-half convention: (x1, x2) -> (x1*cos - x2*sin, x2*cos +
// x1*sin), where x1 is the first rotPairs dims and x2 the next rotPairs dims.
func (m *MRoPE) applyToHead(vec []float32, token int) {
	base := token * m.rotPairs
	for i := 0; i < m.rotPairs; i++ {
		x1 := vec[i]
		x2 := vec[m.rotPairs+i]
		c := m.cos[base+i]
		s := m.sin[base+i]
		vec[i] = x1*c - x2*s
		vec[m.rotPairs+i] = x2*c + x1*s
	}
}

// modulate computes x_norm * (1+scale) + shift over emb dims for one token.
func modulate(dst, x, shift, scale []float32) {
	for i := range dst {
		dst[i] = x[i]*(1+scale[i]) + shift[i]
	}
}

// ForwardLayer applies one DiT block to hidden states in place. hidden is
// [tokens, emb] row-major. cond is the adaLN conditioning vector of length
// AdaLNDim. rope must be built for the same token grid.
//
// Block structure (gate-less adaLN, matching the 4*emb block / 2*emb final
// modulation contract):
//
//	h = h + O(Attn(modulate(LN(h), shift_a, scale_a)))
//	h = h + W2(SiLU(W1(modulate(LN(h), shift_m, scale_m))) * W3(...))
func (l DiTLayer) ForwardLayer(cfg Config, hidden []float32, cond []float32, rope *MRoPE) error {
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
	// adaLN modulation parameters from conditioning.
	mod := make([]float32, 4*emb)
	if err := l.AdaLN.Apply(cond, mod); err != nil {
		return err
	}
	shiftA, scaleA := mod[0:emb], mod[emb:2*emb]
	shiftM, scaleM := mod[2*emb:3*emb], mod[3*emb:4*emb]

	eps := float32(cfg.NormEps)
	if eps <= 0 {
		eps = 1e-6
	}
	heads, headDim := cfg.NumHeads, cfg.HeadDim
	scaleAttn := float32(1 / math.Sqrt(float64(headDim)))

	// ---- Attention sublayer ----
	normed := make([]float32, emb)
	modBuf := make([]float32, emb)
	q := make([]float32, tokens*emb)
	k := make([]float32, tokens*emb)
	v := make([]float32, tokens*emb)
	qkv := make([]float32, 3*emb)
	for t := 0; t < tokens; t++ {
		row := hidden[t*emb : (t+1)*emb]
		layerNormTo(normed, row, eps)
		modulate(modBuf, normed, shiftA, scaleA)
		if err := l.QKV.Apply(modBuf, qkv); err != nil {
			return err
		}
		copy(q[t*emb:(t+1)*emb], qkv[0:emb])
		copy(k[t*emb:(t+1)*emb], qkv[emb:2*emb])
		copy(v[t*emb:(t+1)*emb], qkv[2*emb:3*emb])
	}
	// apply MRoPE per head to q,k.
	for t := 0; t < tokens; t++ {
		for h := 0; h < heads; h++ {
			off := t*emb + h*headDim
			rope.applyToHead(q[off:off+headDim], t)
			rope.applyToHead(k[off:off+headDim], t)
		}
	}
	// full (non-causal) scaled dot-product attention per head.
	attnOut := make([]float32, tokens*emb)
	scores := make([]float32, tokens)
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
			if !simd.SoftmaxInPlace(scores) {
				softmaxFallback(scores)
			}
			ooff := ti*emb + h*headDim
			for tj := 0; tj < tokens; tj++ {
				w := scores[tj]
				voff := tj*emb + h*headDim
				for d := 0; d < headDim; d++ {
					attnOut[ooff+d] += w * v[voff+d]
				}
			}
		}
	}
	// output projection + residual.
	oproj := make([]float32, emb)
	for t := 0; t < tokens; t++ {
		if err := l.O.Apply(attnOut[t*emb:(t+1)*emb], oproj); err != nil {
			return err
		}
		row := hidden[t*emb : (t+1)*emb]
		for i := 0; i < emb; i++ {
			row[i] += oproj[i]
		}
	}

	// ---- MLP sublayer (SwiGLU) ----
	inter := cfg.IntermediateSize
	g := make([]float32, inter)
	u := make([]float32, inter)
	down := make([]float32, emb)
	for t := 0; t < tokens; t++ {
		row := hidden[t*emb : (t+1)*emb]
		layerNormTo(normed, row, eps)
		modulate(modBuf, normed, shiftM, scaleM)
		if err := l.W1.Apply(modBuf, g); err != nil {
			return err
		}
		if err := l.W3.Apply(modBuf, u); err != nil {
			return err
		}
		for i := 0; i < inter; i++ {
			g[i] = siluScalar(g[i]) * u[i]
		}
		if err := l.W2.Apply(g, down); err != nil {
			return err
		}
		for i := 0; i < emb; i++ {
			row[i] += down[i]
		}
	}
	return nil
}

// layerNormTo computes a non-affine LayerNorm (mean/var over the row) into dst.
func layerNormTo(dst, x []float32, eps float32) {
	n := len(x)
	var mean float32
	for _, v := range x {
		mean += v
	}
	mean /= float32(n)
	var variance float32
	for _, v := range x {
		d := v - mean
		variance += d * d
	}
	variance /= float32(n)
	inv := float32(1 / math.Sqrt(float64(variance)+float64(eps)))
	for i := 0; i < n; i++ {
		dst[i] = (x[i] - mean) * inv
	}
}

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
