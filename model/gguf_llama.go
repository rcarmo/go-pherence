// GGUFLlama is a pure-Go LLaMA forward pass loaded from a GGUF file.
// Hot-path linear ops are routed through a k3.OpBackend (CPU SIMD / Vulkan /
// SpacemiT ORT), selected at load time by the caller.
package model

import (
	"fmt"
	"math"

	"github.com/rcarmo/go-pherence/backends/k3"
	"github.com/rcarmo/go-pherence/loader/gguf"
)

// GGUFLlamaConfig holds the hyper-parameters extracted from GGUF metadata.
type GGUFLlamaConfig struct {
	HiddenSize    int
	NumLayers     int
	NumHeads      int // query heads
	NumKVHeads    int
	HeadDim       int
	FFNHiddenSize int
	VocabSize     int
	MaxSeqLen     int
	RMSNormEps    float32
	RopeFreqBase  float32
	RopeDimCount  int
}

// GGUFLlamaLayer holds per-layer weight matrices.
type GGUFLlamaLayer struct {
	AttnNorm []float32 // [hidden]
	FFNNorm  []float32 // [hidden]
	WQ       []float32 // [outDim=hidden, inDim=hidden] row-major
	WK       []float32 // [outDim=kvDim, inDim=hidden]
	WV       []float32 // [outDim=kvDim, inDim=hidden]
	WO       []float32 // [outDim=hidden, inDim=hidden]
	WGate    []float32 // [outDim=ffn, inDim=hidden]
	WUp      []float32 // [outDim=ffn, inDim=hidden]
	WDown    []float32 // [outDim=hidden, inDim=ffn]
}

// GGUFLlama is a loaded LLaMA model with all weights dequanted to F32.
type GGUFLlama struct {
	Config      GGUFLlamaConfig
	Layers      []GGUFLlamaLayer
	EmbedTokens []float32 // [vocab × hidden]
	OutputNorm  []float32 // [hidden]
	LMHead      []float32 // [vocab × hidden]
	Backend     k3.OpBackend
	// precomputed RoPE frequencies [maxSeqLen × rotHalf]
	ropeFreqs []float32
	rotHalf   int
}

// LoadGGUFLlama opens path, reads config, dequants all weights, precomputes
// RoPE frequencies, and returns a ready-to-use GGUFLlama.
func LoadGGUFLlama(path string, backend k3.OpBackend) (*GGUFLlama, error) {
	g, err := gguf.Open(path)
	if err != nil {
		return nil, fmt.Errorf("LoadGGUFLlama: open: %w", err)
	}
	defer g.Close()

	cfg, err := ggufParseConfig(g)
	if err != nil {
		return nil, fmt.Errorf("LoadGGUFLlama: config: %w", err)
	}

	load := func(name string) ([]float32, error) {
		t, ok := g.TensorByName(name)
		if !ok {
			return nil, fmt.Errorf("tensor %q not found", name)
		}
		return g.DequantF32(t)
	}

	embedTokens, err := load("token_embd.weight")
	if err != nil {
		return nil, err
	}
	outputNorm, err := load("output_norm.weight")
	if err != nil {
		return nil, err
	}
	lmHead, err := load("output.weight")
	if err != nil {
		return nil, err
	}

	layers := make([]GGUFLlamaLayer, cfg.NumLayers)
	for i := range layers {
		p := fmt.Sprintf("blk.%d.", i)
		var layer GGUFLlamaLayer
		for dst, suffix := range map[*[]float32]string{
			&layer.AttnNorm: "attn_norm.weight",
			&layer.FFNNorm:  "ffn_norm.weight",
			&layer.WQ:       "attn_q.weight",
			&layer.WK:       "attn_k.weight",
			&layer.WV:       "attn_v.weight",
			&layer.WO:       "attn_output.weight",
			&layer.WGate:    "ffn_gate.weight",
			&layer.WUp:      "ffn_up.weight",
			&layer.WDown:    "ffn_down.weight",
		} {
			data, err := load(p + suffix)
			if err != nil {
				return nil, fmt.Errorf("layer %d %s: %w", i, suffix, err)
			}
			*dst = data
		}
		layers[i] = layer
	}

	m := &GGUFLlama{
		Config:      cfg,
		Layers:      layers,
		EmbedTokens: embedTokens,
		OutputNorm:  outputNorm,
		LMHead:      lmHead,
		Backend:     backend,
	}
	m.precomputeRoPE()
	return m, nil
}

// ggufParseConfig extracts LLaMA config from GGUF metadata.
func ggufParseConfig(g *gguf.GGUF) (GGUFLlamaConfig, error) {
	req := func(key string) (uint32, error) {
		v, ok := g.MetaUint32(key)
		if !ok {
			return 0, fmt.Errorf("missing metadata key %q", key)
		}
		return v, nil
	}
	hidden, err := req("llama.embedding_length")
	if err != nil {
		return GGUFLlamaConfig{}, err
	}
	layers, err := req("llama.block_count")
	if err != nil {
		return GGUFLlamaConfig{}, err
	}
	heads, err := req("llama.attention.head_count")
	if err != nil {
		return GGUFLlamaConfig{}, err
	}
	kvHeads, err := req("llama.attention.head_count_kv")
	if err != nil {
		return GGUFLlamaConfig{}, err
	}
	ffn, err := req("llama.feed_forward_length")
	if err != nil {
		return GGUFLlamaConfig{}, err
	}
	maxCtx, err := req("llama.context_length")
	if err != nil {
		return GGUFLlamaConfig{}, err
	}
	vocabSize, err := req("llama.vocab_size")
	if err != nil {
		// fallback: count from token_embd shape
		vocabSize = 32000
	}
	ropeBase := float32(10000.0)
	if v, ok := g.MetaFloat32("llama.rope.freq_base"); ok {
		ropeBase = v
	}
	eps := float32(1e-5)
	if v, ok := g.MetaFloat32("llama.attention.layer_norm_rms_epsilon"); ok {
		eps = v
	}
	ropeDim := int(hidden) / int(heads)
	if v, ok := g.MetaUint32("llama.rope.dimension_count"); ok {
		ropeDim = int(v)
	}
	cfg := GGUFLlamaConfig{
		HiddenSize:    int(hidden),
		NumLayers:     int(layers),
		NumHeads:      int(heads),
		NumKVHeads:    int(kvHeads),
		HeadDim:       int(hidden) / int(heads),
		FFNHiddenSize: int(ffn),
		VocabSize:     int(vocabSize),
		MaxSeqLen:     int(maxCtx),
		RMSNormEps:    eps,
		RopeFreqBase:  ropeBase,
		RopeDimCount:  ropeDim,
	}
	return cfg, nil
}

// precomputeRoPE precomputes [maxSeqLen × rotHalf] cos/sin interleaved frequencies.
// We store them flat as [pos × rotHalf] complex rotations encoded as (cos,sin) pairs.
func (m *GGUFLlama) precomputeRoPE() {
	cfg := m.Config
	rotHalf := cfg.RopeDimCount / 2
	m.rotHalf = rotHalf
	m.ropeFreqs = make([]float32, cfg.MaxSeqLen*rotHalf)
	for pos := 0; pos < cfg.MaxSeqLen; pos++ {
		for i := 0; i < rotHalf; i++ {
			theta := float64(pos) / math.Pow(float64(cfg.RopeFreqBase), float64(2*i)/float64(cfg.RopeDimCount))
			m.ropeFreqs[pos*rotHalf+i] = float32(theta)
		}
	}
}

// ── core math helpers ─────────────────────────────────────────────────────────

func rmsNormF32Inplace(x, w []float32, eps float32) {
	var sum float32
	for _, v := range x {
		sum += v * v
	}
	scale := float32(1.0 / math.Sqrt(float64(sum/float32(len(x))+eps)))
	for i := range x {
		x[i] = w[i] * x[i] * scale
	}
}

// applyRoPEInplace applies RoPE to a Q or K vector in-place.
// x is [nHeads × headDim]; freqs is [rotHalf] (the pre-computed theta values for this position).
func applyRoPEInplace(x []float32, freqs []float32, nHeads, headDim, rotHalf int) {
	for h := 0; h < nHeads; h++ {
		row := x[h*headDim : (h+1)*headDim]
		for i := 0; i < rotHalf; i++ {
			theta := freqs[i]
			cos := float32(math.Cos(float64(theta)))
			sin := float32(math.Sin(float64(theta)))
			x0 := row[i]
			x1 := row[i+rotHalf]
			row[i] = x0*cos - x1*sin
			row[i+rotHalf] = x0*sin + x1*cos
		}
	}
}

func softmaxInplace(x []float32) {
	max := x[0]
	for _, v := range x[1:] {
		if v > max {
			max = v
		}
	}
	var sum float32
	for i, v := range x {
		x[i] = float32(math.Exp(float64(v - max)))
		sum += x[i]
	}
	inv := float32(1.0 / float64(sum))
	for i := range x {
		x[i] *= inv
	}
}

// ── backend-dispatched GEMV ───────────────────────────────────────────────────

func (m *GGUFLlama) gemv(out, x, w []float32, inDim, outDim int) {
	if err := m.Backend.GemvF32(out, x, w, inDim, outDim); err != nil {
		// hard fallback: plain dot
		for i := 0; i < outDim; i++ {
			var sum float32
			row := w[i*inDim : (i+1)*inDim]
			for j, xv := range x {
				sum += row[j] * xv
			}
			out[i] = sum
		}
	}
}

func (m *GGUFLlama) rmsNorm(x, w []float32) {
	if err := m.Backend.RMSNormF32(x, w, m.Config.RMSNormEps); err != nil {
		rmsNormF32Inplace(x, w, m.Config.RMSNormEps)
	}
}

func (m *GGUFLlama) siluMul(dst, gate, up []float32) {
	if err := m.Backend.SiLUMulF32(dst, gate, up); err != nil {
		// scalar fallback
		for i := range gate {
			g := gate[i]
			silu := g * float32(1.0/(1.0+math.Exp(float64(-g))))
			dst[i] = silu * up[i]
		}
	}
}

// ── forward pass ─────────────────────────────────────────────────────────────

// Forward runs a single token through the model, updating the KV cache,
// and returns the logits vector [vocabSize].
//
// kvK[layer][step*kvDim : (step+1)*kvDim] and kvV[...] are the KV caches.
func (m *GGUFLlama) Forward(tokenID, step int, kvK, kvV [][]float32) []float32 {
	cfg := m.Config
	h := cfg.HiddenSize
	nH := cfg.NumHeads
	nKV := cfg.NumKVHeads
	hDim := cfg.HeadDim
	kvDim := nKV * hDim
	ffn := cfg.FFNHiddenSize
	rotHalf := m.rotHalf

	// Token embedding
	hidden := make([]float32, h)
	copy(hidden, m.EmbedTokens[tokenID*h:(tokenID+1)*h])

	// RoPE frequencies for this position
	posFreqs := m.ropeFreqs[step*rotHalf : (step+1)*rotHalf]

	// Reusable per-token scratch buffers. These used to be allocated per layer,
	// which created hundreds of short-lived slices per generated token.
	attnIn := make([]float32, h)
	q := make([]float32, nH*hDim)
	k := make([]float32, kvDim)
	v := make([]float32, kvDim)
	attnOut := make([]float32, nH*hDim)
	attnScores := make([]float32, cfg.MaxSeqLen)
	oOut := make([]float32, h)
	ffnIn := make([]float32, h)
	gate := make([]float32, ffn)
	up := make([]float32, ffn)
	ffnMid := make([]float32, ffn)
	down := make([]float32, h)

	for i, layer := range m.Layers {
		// ── attention sub-layer ───────────────────────────────────────────
		copy(attnIn, hidden)
		m.rmsNorm(attnIn, layer.AttnNorm)
		m.gemv(q, attnIn, layer.WQ, h, nH*hDim)
		m.gemv(k, attnIn, layer.WK, h, kvDim)
		m.gemv(v, attnIn, layer.WV, h, kvDim)

		// RoPE
		applyRoPEInplace(q, posFreqs, nH, hDim, rotHalf)
		applyRoPEInplace(k, posFreqs, nKV, hDim, rotHalf)

		// Update KV cache
		kCache := kvK[i]
		vCache := kvV[i]
		copy(kCache[step*kvDim:], k)
		copy(vCache[step*kvDim:], v)

		// Grouped-query attention: compute attention output
		m.gqaAttentionInto(attnOut, attnScores, q, kCache, vCache, step+1, nH, nKV, hDim)

		// Output projection
		m.gemv(oOut, attnOut, layer.WO, nH*hDim, h)

		// Residual add
		for j := range hidden {
			hidden[j] += oOut[j]
		}

		// ── FFN sub-layer ─────────────────────────────────────────────────
		copy(ffnIn, hidden)
		m.rmsNorm(ffnIn, layer.FFNNorm)

		m.gemv(gate, ffnIn, layer.WGate, h, ffn)
		m.gemv(up, ffnIn, layer.WUp, h, ffn)

		m.siluMul(ffnMid, gate, up)

		m.gemv(down, ffnMid, layer.WDown, ffn, h)

		for j := range hidden {
			hidden[j] += down[j]
		}
	}

	// Final norm + LM head
	m.rmsNorm(hidden, m.OutputNorm)
	logits := make([]float32, cfg.VocabSize)
	m.gemv(logits, hidden, m.LMHead, h, cfg.VocabSize)
	return logits
}

// gqaAttention computes multi-head attention with GQA.
// q: [nH × hDim], kCache: [seqLen × kvDim], vCache: [seqLen × kvDim]
// Returns attention output [nH × hDim].
func (m *GGUFLlama) gqaAttentionInto(out, scores, q, kCache, vCache []float32, seqLen, nH, nKV, hDim int) {
	scale := float32(1.0 / math.Sqrt(float64(hDim)))
	groupSize := nH / nKV
	kvDim := nKV * hDim
	for i := 0; i < nH*hDim; i++ {
		out[i] = 0
	}

	for h := 0; h < nH; h++ {
		kvHead := h / groupSize
		qRow := q[h*hDim : (h+1)*hDim]

		for t := 0; t < seqLen; t++ {
			kRow := kCache[t*kvDim+kvHead*hDim : t*kvDim+(kvHead+1)*hDim]
			var dot float32
			for d := 0; d < hDim; d++ {
				dot += qRow[d] * kRow[d]
			}
			scores[t] = dot * scale
		}
		softmaxInplace(scores[:seqLen])

		outRow := out[h*hDim : (h+1)*hDim]
		for t := 0; t < seqLen; t++ {
			vRow := vCache[t*kvDim+kvHead*hDim : t*kvDim+(kvHead+1)*hDim]
			w := scores[t]
			for d := 0; d < hDim; d++ {
				outRow[d] += w * vRow[d]
			}
		}
	}
}

// gqaAttention computes multi-head attention with GQA.
// q: [nH × hDim], kCache: [seqLen × kvDim], vCache: [seqLen × kvDim]
// Returns attention output [nH × hDim].
func (m *GGUFLlama) gqaAttention(q, kCache, vCache []float32, seqLen, nH, nKV, hDim int) []float32 {
	out := make([]float32, nH*hDim)
	scores := make([]float32, seqLen)
	m.gqaAttentionInto(out, scores, q, kCache, vCache, seqLen, nH, nKV, hDim)
	return out
}

// Generate runs autoregressive generation for up to maxNew tokens.
// Prompt token IDs must already include BOS if required.
// Returns generated token IDs (not including the prompt).
func (m *GGUFLlama) Generate(promptIDs []int, maxNew int) ([]int, error) {
	cfg := m.Config
	kvDim := cfg.NumKVHeads * cfg.HeadDim
	maxSeq := len(promptIDs) + maxNew
	if maxSeq > cfg.MaxSeqLen {
		maxSeq = cfg.MaxSeqLen
	}

	// Allocate KV caches
	kvK := make([][]float32, cfg.NumLayers)
	kvV := make([][]float32, cfg.NumLayers)
	for i := range kvK {
		kvK[i] = make([]float32, maxSeq*kvDim)
		kvV[i] = make([]float32, maxSeq*kvDim)
	}

	var generated []int

	// Prefill: run all prompt tokens
	for step, tok := range promptIDs {
		if step == len(promptIDs)-1 {
			// Last prompt token: capture logits
			logits := m.Forward(tok, step, kvK, kvV)
			next := argmaxF32(logits)
			if next == cfg.VocabSize { // shouldn't happen
				break
			}
		} else {
			_ = m.Forward(tok, step, kvK, kvV)
		}
	}

	// Decode: autoregressively generate maxNew tokens
	step := len(promptIDs) - 1
	logits := m.Forward(promptIDs[step], step, kvK, kvV)
	for range maxNew {
		next := argmaxF32(logits)
		generated = append(generated, next)
		if next == cfg.VocabSize-1 || (cfg.VocabSize > 2 && next == 2) {
			// EOS
			break
		}
		step++
		if step >= maxSeq {
			break
		}
		logits = m.Forward(next, step, kvK, kvV)
	}
	return generated, nil
}

// argmaxF32 returns the index of the maximum value in x.
func argmaxF32(x []float32) int {
	best := 0
	for i, v := range x[1:] {
		if v > x[best] {
			best = i + 1
		}
	}
	return best
}
