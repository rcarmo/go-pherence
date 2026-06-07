package ideogram4

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/rcarmo/go-pherence/half"
)

// CombinedTensorSource exposes both raw and float32 tensor access, satisfied by
// *safetensors.File and *safetensors.ShardedFile.
type CombinedTensorSource interface {
	GetRaw(name string) ([]byte, string, []int, error)
	GetFloat32(name string) ([]float32, []int, error)
}

// QwenVLConditioner is the native Qwen3-VL text-only conditioning forward. It
// runs the language-model stack over a token sequence and concatenates the
// hidden states at Config.ActivationLayers into the [tokens, llm_features_dim]
// tensor consumed by the Ideogram4 DiT.
//
// hidden_states indexing follows HF convention: index 0 is the post-embedding
// state, index k (1..num_layers) is the residual stream after decoder layer
// k-1.
type QwenVLConditioner struct {
	src     CombinedTensorSource
	cfg     Config
	prefix  string
	theta   float64
	eps     float32
	embed   []byte // raw bf16 [vocab, hidden]
	embedSh []int
}

// NewQwenVLConditioner binds the conditioner to the text-encoder weight source.
// namePrefix is the tensor-name prefix (e.g. "language_model").
func NewQwenVLConditioner(src CombinedTensorSource, cfg Config, namePrefix string) (*QwenVLConditioner, error) {
	if src == nil {
		return nil, fmt.Errorf("ideogram4 qwen-vl: nil source")
	}
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if cfg.TextHeads <= 0 || cfg.TextKVHeads <= 0 || cfg.TextHeadDim <= 0 || cfg.TextIntermediate <= 0 {
		return nil, fmt.Errorf("ideogram4 qwen-vl: incomplete text dims %+v", cfg)
	}
	theta := cfg.TextRopeTheta
	if theta <= 0 {
		theta = 1000000
	}
	eps := float32(cfg.TextRmsEps)
	if eps <= 0 {
		eps = 1e-6
	}
	embed, _, sh, err := src.GetRaw(namePrefix + ".embed_tokens.weight")
	if err != nil {
		return nil, fmt.Errorf("ideogram4 qwen-vl embed: %w", err)
	}
	if len(sh) != 2 || sh[1] != cfg.TextHidden {
		return nil, fmt.Errorf("ideogram4 qwen-vl embed shape=%v want [vocab,%d]", sh, cfg.TextHidden)
	}
	return &QwenVLConditioner{src: src, cfg: cfg, prefix: namePrefix, theta: theta, eps: eps, embed: embed, embedSh: sh}, nil
}

func (q *QwenVLConditioner) loadFP8(name string, outDim, inDim int) (*FP8Linear, error) {
	wb, wd, wsh, err := q.src.GetRaw(name + ".weight")
	if err != nil {
		return nil, fmt.Errorf("ideogram4 qwen-vl %q: %w", name, err)
	}
	if wd != "F8_E4M3" && wd != "F8_E4M3FN" {
		return nil, fmt.Errorf("ideogram4 qwen-vl %q dtype=%s want F8", name, wd)
	}
	if len(wsh) != 2 || wsh[0] != outDim || wsh[1] != inDim {
		return nil, fmt.Errorf("ideogram4 qwen-vl %q shape=%v want [%d,%d]", name, wsh, outDim, inDim)
	}
	sb, sd, ssh, err := q.src.GetRaw(name + ".weight_scale")
	if err != nil {
		return nil, fmt.Errorf("ideogram4 qwen-vl %q scale: %w", name, err)
	}
	spec := LinearSpec{Prefix: name, OutDim: outDim, InDim: inDim, Weight: name + ".weight", WeightScale: name + ".weight_scale"}
	scale, err := decodeScale(spec, sb, sd, ssh)
	if err != nil {
		return nil, err
	}
	lin, err := NewFP8Linear(spec, wb, scale, nil)
	if err != nil {
		return nil, fmt.Errorf("ideogram4 qwen-vl %q: %w", name, err)
	}
	return lin, nil
}

func (q *QwenVLConditioner) rmsWeight(name string, n int) ([]float32, error) {
	w, sh, err := q.src.GetFloat32(name)
	if err != nil {
		return nil, fmt.Errorf("ideogram4 qwen-vl norm %q: %w", name, err)
	}
	total := 1
	for _, d := range sh {
		total *= d
	}
	if total != n {
		return nil, fmt.Errorf("ideogram4 qwen-vl norm %q len=%d want=%d", name, total, n)
	}
	return w, nil
}

// embedToken decodes one embedding row (bf16) into dst.
func (q *QwenVLConditioner) embedToken(id int, dst []float32) error {
	hidden := q.cfg.TextHidden
	if id < 0 || id >= q.embedSh[0] {
		return fmt.Errorf("ideogram4 qwen-vl token id=%d vocab=%d", id, q.embedSh[0])
	}
	base := id * hidden * 2
	for i := 0; i < hidden; i++ {
		dst[i] = half.BF16ToF32(binary.LittleEndian.Uint16(q.embed[base+i*2:]))
	}
	return nil
}

// Condition runs the Qwen3-VL stack over tokenIDs and returns the concatenated
// activation-layer hidden states: [len(tokenIDs), llm_features_dim].
func (q *QwenVLConditioner) Condition(tokenIDs []int) ([]float32, error) {
	if q == nil {
		return nil, ErrRuntimeNotImplemented
	}
	if len(tokenIDs) == 0 {
		return nil, fmt.Errorf("ideogram4 qwen-vl: empty token sequence")
	}
	cfg := q.cfg
	hidden := cfg.TextHidden
	T := len(tokenIDs)
	heads, kvHeads, headDim := cfg.TextHeads, cfg.TextKVHeads, cfg.TextHeadDim
	group := heads / kvHeads
	if heads%kvHeads != 0 {
		return nil, fmt.Errorf("ideogram4 qwen-vl heads=%d kv=%d not divisible", heads, kvHeads)
	}
	qDim, kvDim := heads*headDim, kvHeads*headDim

	// residual stream [T, hidden].
	h := make([]float32, T*hidden)
	for t := 0; t < T; t++ {
		if err := q.embedToken(tokenIDs[t], h[t*hidden:(t+1)*hidden]); err != nil {
			return nil, err
		}
	}

	// capture hidden states at activation layers.
	want := make(map[int]bool, len(cfg.ActivationLayers))
	for _, l := range cfg.ActivationLayers {
		want[l] = true
	}
	captured := make(map[int][]float32, len(cfg.ActivationLayers))
	if want[0] {
		captured[0] = append([]float32(nil), h...)
	}

	rope := buildRoPECosSin(T, headDim, q.theta)
	tmp := make([]float32, hidden)
	for layer := 0; layer < cfg.TextLayers; layer++ {
		lp := fmt.Sprintf("%s.layers.%d", q.prefix, layer)
		if err := q.decoderLayer(h, T, lp, heads, kvHeads, headDim, group, qDim, kvDim, rope, tmp); err != nil {
			return nil, fmt.Errorf("ideogram4 qwen-vl layer %d: %w", layer, err)
		}
		if want[layer+1] {
			captured[layer+1] = append([]float32(nil), h...)
		}
	}

	// concatenate captured states in activation-layer order.
	feat := cfg.LLMFeaturesDim
	out := make([]float32, T*feat)
	for t := 0; t < T; t++ {
		off := 0
		for _, l := range cfg.ActivationLayers {
			capState, ok := captured[l]
			if !ok {
				return nil, fmt.Errorf("ideogram4 qwen-vl missing activation layer %d", l)
			}
			copy(out[t*feat+off:t*feat+off+hidden], capState[t*hidden:(t+1)*hidden])
			off += hidden
		}
	}
	return out, nil
}

func (q *QwenVLConditioner) decoderLayer(h []float32, T int, lp string, heads, kvHeads, headDim, group, qDim, kvDim int, rope ropeTable, tmp []float32) error {
	hidden := q.cfg.TextHidden
	inW, err := q.rmsWeight(lp+".input_layernorm.weight", hidden)
	if err != nil {
		return err
	}
	qProj, err := q.loadFP8(lp+".self_attn.q_proj", qDim, hidden)
	if err != nil {
		return err
	}
	kProj, err := q.loadFP8(lp+".self_attn.k_proj", kvDim, hidden)
	if err != nil {
		return err
	}
	vProj, err := q.loadFP8(lp+".self_attn.v_proj", kvDim, hidden)
	if err != nil {
		return err
	}
	oProj, err := q.loadFP8(lp+".self_attn.o_proj", hidden, qDim)
	if err != nil {
		return err
	}
	defer qProj.ReleaseGPU()
	defer kProj.ReleaseGPU()
	defer vProj.ReleaseGPU()
	defer oProj.ReleaseGPU()
	qNorm, err := q.rmsWeight(lp+".self_attn.q_norm.weight", headDim)
	if err != nil {
		return err
	}
	kNorm, err := q.rmsWeight(lp+".self_attn.k_norm.weight", headDim)
	if err != nil {
		return err
	}

	qh := make([]float32, T*qDim)
	kh := make([]float32, T*kvDim)
	vh := make([]float32, T*kvDim)
	for t := 0; t < T; t++ {
		rmsNormTo(tmp, h[t*hidden:(t+1)*hidden], inW, q.eps)
		if err := qProj.Apply(tmp, qh[t*qDim:(t+1)*qDim]); err != nil {
			return err
		}
		if err := kProj.Apply(tmp, kh[t*kvDim:(t+1)*kvDim]); err != nil {
			return err
		}
		if err := vProj.Apply(tmp, vh[t*kvDim:(t+1)*kvDim]); err != nil {
			return err
		}
		// per-head q/k RMSNorm then RoPE.
		for hd := 0; hd < heads; hd++ {
			seg := qh[t*qDim+hd*headDim : t*qDim+(hd+1)*headDim]
			rmsNormInPlace(seg, qNorm, q.eps)
			rope.apply(seg, t)
		}
		for hd := 0; hd < kvHeads; hd++ {
			seg := kh[t*kvDim+hd*headDim : t*kvDim+(hd+1)*headDim]
			rmsNormInPlace(seg, kNorm, q.eps)
			rope.apply(seg, t)
		}
	}

	// causal GQA attention.
	attn := make([]float32, T*qDim)
	scale := float32(1 / math.Sqrt(float64(headDim)))
	qwenCausalGQAAttention(attn, qh, kh, vh, T, heads, kvHeads, headDim, group, qDim, kvDim, scale)
	// output projection + residual.
	oBuf := make([]float32, hidden)
	for t := 0; t < T; t++ {
		if err := oProj.Apply(attn[t*qDim:(t+1)*qDim], oBuf); err != nil {
			return err
		}
		row := h[t*hidden : (t+1)*hidden]
		for i := 0; i < hidden; i++ {
			row[i] += oBuf[i]
		}
	}

	// MLP (SwiGLU).
	postW, err := q.rmsWeight(lp+".post_attention_layernorm.weight", hidden)
	if err != nil {
		return err
	}
	inter := q.cfg.TextIntermediate
	gate, err := q.loadFP8(lp+".mlp.gate_proj", inter, hidden)
	if err != nil {
		return err
	}
	up, err := q.loadFP8(lp+".mlp.up_proj", inter, hidden)
	if err != nil {
		return err
	}
	down, err := q.loadFP8(lp+".mlp.down_proj", hidden, inter)
	if err != nil {
		return err
	}
	defer gate.ReleaseGPU()
	defer up.ReleaseGPU()
	defer down.ReleaseGPU()
	g := make([]float32, inter)
	u := make([]float32, inter)
	dBuf := make([]float32, hidden)
	for t := 0; t < T; t++ {
		rmsNormTo(tmp, h[t*hidden:(t+1)*hidden], postW, q.eps)
		if err := gate.Apply(tmp, g); err != nil {
			return err
		}
		if err := up.Apply(tmp, u); err != nil {
			return err
		}
		for i := 0; i < inter; i++ {
			g[i] = siluScalar(g[i]) * u[i]
		}
		if err := down.Apply(g, dBuf); err != nil {
			return err
		}
		row := h[t*hidden : (t+1)*hidden]
		for i := 0; i < hidden; i++ {
			row[i] += dBuf[i]
		}
	}
	return nil
}

// ---- helpers ----

func qwenCausalGQAAttention(attn, qh, kh, vh []float32, T, heads, kvHeads, headDim, group, qDim, kvDim int, scale float32) {
	if gpuAttentionEnabled() {
		ok := true
		for ti := 0; ti < T; ti++ {
			out := attn[ti*qDim : (ti+1)*qDim]
			qRow := qh[ti*qDim : (ti+1)*qDim]
			kPrefix := kh[:(ti+1)*kvDim]
			vPrefix := vh[:(ti+1)*kvDim]
			if err := qwenGQAAttentionGPU(out, qRow, kPrefix, vPrefix, ti+1, heads, kvHeads, headDim, scale); err != nil {
				if gpuAttentionStrict() {
					panic(err)
				}
				ok = false
				break
			}
		}
		if ok {
			return
		}
		for i := range attn {
			attn[i] = 0
		}
	}
	scores := make([]float32, T)
	for hd := 0; hd < heads; hd++ {
		kvh := hd / group
		for ti := 0; ti < T; ti++ {
			qoff := ti*qDim + hd*headDim
			for tj := 0; tj <= ti; tj++ {
				koff := tj*kvDim + kvh*headDim
				var dot float32
				for d := 0; d < headDim; d++ {
					dot += qh[qoff+d] * kh[koff+d]
				}
				scores[tj] = dot * scale
			}
			softmaxFallback(scores[:ti+1])
			ooff := ti*qDim + hd*headDim
			for tj := 0; tj <= ti; tj++ {
				w := scores[tj]
				voff := tj*kvDim + kvh*headDim
				for d := 0; d < headDim; d++ {
					attn[ooff+d] += w * vh[voff+d]
				}
			}
		}
	}
}

func rmsNormTo(dst, x, weight []float32, eps float32) {
	if gpuNormEnabled() {
		if err := rmsNormWeightedGPU(dst, x, weight, eps); err == nil {
			return
		} else if gpuNormStrict() {
			panic(err)
		}
	}
	rmsNormToCPU(dst, x, weight, eps)
}

func rmsNormToCPU(dst, x, weight []float32, eps float32) {
	var ss float64
	for _, v := range x {
		ss += float64(v) * float64(v)
	}
	inv := float32(1 / math.Sqrt(ss/float64(len(x))+float64(eps)))
	for i := range x {
		dst[i] = x[i] * inv * weight[i]
	}
}

func rmsNormInPlace(x, weight []float32, eps float32) {
	if gpuNormEnabled() {
		tmp := make([]float32, len(x))
		if err := rmsNormWeightedGPU(tmp, x, weight, eps); err == nil {
			copy(x, tmp)
			return
		} else if gpuNormStrict() {
			panic(err)
		}
	}
	rmsNormToCPU(x, x, weight, eps)
}

type ropeTable struct {
	headDim int
	half    int
	cos     []float32 // [T, half]
	sin     []float32
}

func buildRoPECosSin(T, headDim int, theta float64) ropeTable {
	half := headDim / 2
	rt := ropeTable{headDim: headDim, half: half, cos: make([]float32, T*half), sin: make([]float32, T*half)}
	for t := 0; t < T; t++ {
		for i := 0; i < half; i++ {
			freq := math.Pow(theta, -float64(2*i)/float64(headDim))
			ang := float64(t) * freq
			rt.cos[t*half+i] = float32(math.Cos(ang))
			rt.sin[t*half+i] = float32(math.Sin(ang))
		}
	}
	return rt
}

// apply rotates a head vector with the rotate-half (NeoX) convention.
func (rt ropeTable) apply(vec []float32, t int) {
	if gpuMRoPEEnabled() {
		if err := qwenRoPEGPU(vec, rt, t); err == nil {
			return
		} else if gpuMRoPEStrict() {
			panic(err)
		}
	}
	base := t * rt.half
	for i := 0; i < rt.half; i++ {
		x1 := vec[i]
		x2 := vec[rt.half+i]
		c := rt.cos[base+i]
		s := rt.sin[base+i]
		vec[i] = x1*c - x2*s
		vec[rt.half+i] = x2*c + x1*s
	}
}
