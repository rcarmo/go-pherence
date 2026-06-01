// GGUFLlama is a pure-Go LLaMA forward pass loaded from a GGUF file.
// Hot-path linear ops are routed through a k3.OpBackend (CPU SIMD / Vulkan /
// SpacemiT ORT), selected at load time by the caller.
package model

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sync"

	"github.com/rcarmo/go-pherence/backends/ggmlgraph"
	"github.com/rcarmo/go-pherence/backends/ggmlquant"
	"github.com/rcarmo/go-pherence/backends/k3"
	"github.com/rcarmo/go-pherence/loader/gguf"
	gograph "github.com/rcarmo/go-pherence/runtime/graph"
	"github.com/rcarmo/go-pherence/runtime/kv"
)

// GGUFLlamaConfig holds the hyper-parameters extracted from GGUF metadata.
type GGUFLlamaConfig struct {
	Architecture          string
	HiddenSize            int
	NumLayers             int
	NumHeads              int // query heads
	NumKVHeads            int
	HeadDim               int
	FFNHiddenSize         int
	VocabSize             int
	MaxSeqLen             int
	RMSNormEps            float32
	RopeFreqBase          float32
	RopeDimCount          int
	NumExperts            int
	NumExpertsPerTok      int
	MoEHiddenSize         int
	SharedMoEHiddenSize   int
	FullAttentionInterval int
	AttentionKeyLength    int
	AttentionValueLength  int
	SSMConvKernel         int
	SSMGroupCount         int
	SSMInnerSize          int
	SSMStateSize          int
	SSMTimeStepRank       int
	BOSTokenID            int
	EOSTokenID            int
}

// GGUFLlamaLayer holds per-layer weight matrices.
type GGUFLlamaLayer struct {
	AttnNorm []float32 // [hidden]
	FFNNorm  []float32 // [hidden]
	WQ       []float32 // [outDim=hidden, inDim=hidden] row-major
	WQm      *gguf.QuantMatrix
	WK       []float32 // [outDim=kvDim, inDim=hidden]
	WKm      *gguf.QuantMatrix
	WV       []float32 // [outDim=kvDim, inDim=hidden]
	WVm      *gguf.QuantMatrix
	WO       []float32 // [outDim=hidden, inDim=hidden]
	WOm      *gguf.QuantMatrix
	WGate    []float32 // [outDim=ffn, inDim=hidden]
	WGateM   *gguf.QuantMatrix
	WUp      []float32 // [outDim=ffn, inDim=hidden]
	WUpM     *gguf.QuantMatrix
	WDown    []float32 // [outDim=hidden, inDim=ffn]
	WDownM   *gguf.QuantMatrix

	RouterW       []float32 // [outDim=experts, inDim=hidden]
	RouterM       *gguf.QuantMatrix
	ExpertGateM   *gguf.ExpertMatrices
	ExpertUpM     *gguf.ExpertMatrices
	ExpertDownM   *gguf.ExpertMatrices
	SharedGateM   *gguf.QuantMatrix
	SharedUpM     *gguf.QuantMatrix
	SharedDownM   *gguf.QuantMatrix
	SharedGateInp []float32
	QNorm         []float32
	KNorm         []float32

	FusedQKVM *gguf.QuantMatrix
	AttnGateM *gguf.QuantMatrix
	SSMOutM   *gguf.QuantMatrix
	SSMConv1D []float32
	SSMA      []float32
	SSMDTBias []float32
	SSMNorm   []float32
	SSMAlphaM *gguf.QuantMatrix
	SSMBetaM  *gguf.QuantMatrix
}

// GGUFLlama is a loaded LLaMA model with all weights dequanted to F32.
type GGUFLlama struct {
	Config       GGUFLlamaConfig
	Layers       []GGUFLlamaLayer
	EmbedTokens  []float32 // [vocab × hidden]
	EmbedMatrix  *gguf.QuantMatrix
	OutputNorm   []float32 // [hidden]
	LMHead       []float32 // [vocab × hidden]
	LMHeadM      *gguf.QuantMatrix
	LMHeadGraph  *ggmlgraph.MulMat
	DecodeGraph  *gograph.Graph
	DecodePlan   *gograph.Plan
	UseGGMLQuant bool
	Backend      k3.OpBackend
	REAP         *REAPConfig
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
	if err := cfg.ValidateRuntimeSupported(); err != nil {
		return nil, fmt.Errorf("LoadGGUFLlama: %w", err)
	}

	load := func(name string) ([]float32, error) {
		t, ok := g.TensorByName(name)
		if !ok {
			return nil, fmt.Errorf("tensor %q not found", name)
		}
		return g.DequantF32(t)
	}
	loadMatrix := func(name string) (*gguf.QuantMatrix, error) {
		t, ok := g.TensorByName(name)
		if !ok {
			return nil, fmt.Errorf("tensor %q not found", name)
		}
		return g.MatrixFromTensor(t)
	}
	useGGMLQuant := os.Getenv("GO_PHERENCE_GGML_QUANT") == "1"
	useLMHeadGraph := os.Getenv("GO_PHERENCE_GGML_LMHEAD_GRAPH") == "1"

	var embedTokens []float32
	if !useGGMLQuant {
		embedTokens, err = load("token_embd.weight")
		if err != nil {
			return nil, err
		}
	}
	outputNorm, err := load("output_norm.weight")
	if err != nil {
		return nil, err
	}
	var lmHead []float32
	if !useGGMLQuant && !useLMHeadGraph {
		lmHead, err = load("output.weight")
		if err != nil {
			return nil, err
		}
	}
	var embedMatrix, lmHeadM *gguf.QuantMatrix
	var lmHeadGraph *ggmlgraph.MulMat
	if useGGMLQuant || useLMHeadGraph {
		if lmHeadM, err = loadMatrix("output.weight"); err != nil {
			return nil, err
		}
	}
	if useGGMLQuant {
		if embedMatrix, err = loadMatrix("token_embd.weight"); err != nil {
			return nil, err
		}
	}
	if useLMHeadGraph && lmHeadM != nil {
		if lmHeadGraph, err = ggmlgraph.NewMulMat(int(lmHeadM.QType), lmHeadM.Raw, lmHeadM.InDim, lmHeadM.OutDim, 8); err != nil {
			return nil, err
		}
	}

	layers := make([]GGUFLlamaLayer, cfg.NumLayers)
	for i := range layers {
		p := fmt.Sprintf("blk.%d.", i)
		var layer GGUFLlamaLayer
		baseLoads := map[*[]float32]string{
			&layer.AttnNorm: "attn_norm.weight",
		}
		if cfg.IsQwenNextHybridGGUF() {
			baseLoads[&layer.FFNNorm] = "post_attention_norm.weight"
			if _, ok := g.TensorByName(p + "attn_q.weight"); ok {
				baseLoads[&layer.WQ] = "attn_q.weight"
				baseLoads[&layer.WK] = "attn_k.weight"
				baseLoads[&layer.WV] = "attn_v.weight"
				baseLoads[&layer.WO] = "attn_output.weight"
			}
		} else {
			baseLoads[&layer.FFNNorm] = "ffn_norm.weight"
			baseLoads[&layer.WQ] = "attn_q.weight"
			baseLoads[&layer.WK] = "attn_k.weight"
			baseLoads[&layer.WV] = "attn_v.weight"
			baseLoads[&layer.WO] = "attn_output.weight"
		}
		for dst, suffix := range baseLoads {
			data, err := load(p + suffix)
			if err != nil {
				return nil, fmt.Errorf("layer %d %s: %w", i, suffix, err)
			}
			*dst = data
		}
		if cfg.IsQwenNextHybridGGUF() {
			if err := loadGGUFQwenNextHybridTensors(g, i, &layer); err != nil {
				return nil, fmt.Errorf("layer %d QwenNext hybrid: %w", i, err)
			}
			if qn, err := load(p + "attn_q_norm.weight"); err == nil {
				layer.QNorm = qn
			}
			if kn, err := load(p + "attn_k_norm.weight"); err == nil {
				layer.KNorm = kn
			}
		}
		if cfg.NumExperts > 0 {
			layer.RouterM, layer.ExpertGateM, layer.ExpertUpM, layer.ExpertDownM, err = loadGGUFMoEExpertMatrices(g, i)
			if err != nil {
				return nil, fmt.Errorf("layer %d MoE: %w", i, err)
			}
			layer.SharedGateM, _ = loadMatrix(p + "ffn_gate_shexp.weight")
			layer.SharedUpM, _ = loadMatrix(p + "ffn_up_shexp.weight")
			layer.SharedDownM, _ = loadMatrix(p + "ffn_down_shexp.weight")
			layer.SharedGateInp, _ = load(p + "ffn_gate_inp_shexp.weight")
		} else {
			for dst, suffix := range map[*[]float32]string{
				&layer.WGate: "ffn_gate.weight",
				&layer.WUp:   "ffn_up.weight",
				&layer.WDown: "ffn_down.weight",
			} {
				data, err := load(p + suffix)
				if err != nil {
					return nil, fmt.Errorf("layer %d %s: %w", i, suffix, err)
				}
				*dst = data
			}
		}
		if useGGMLQuant && (!cfg.IsQwenNextHybridGGUF() || layer.WQ != nil) {
			if layer.WQm, err = loadMatrix(p + "attn_q.weight"); err != nil {
				return nil, err
			}
			if layer.WKm, err = loadMatrix(p + "attn_k.weight"); err != nil {
				return nil, err
			}
			if layer.WVm, err = loadMatrix(p + "attn_v.weight"); err != nil {
				return nil, err
			}
			if layer.WOm, err = loadMatrix(p + "attn_output.weight"); err != nil {
				return nil, err
			}
			if cfg.NumExperts == 0 {
				if layer.WGateM, err = loadMatrix(p + "ffn_gate.weight"); err != nil {
					return nil, err
				}
				if layer.WUpM, err = loadMatrix(p + "ffn_up.weight"); err != nil {
					return nil, err
				}
				if layer.WDownM, err = loadMatrix(p + "ffn_down.weight"); err != nil {
					return nil, err
				}
			}
		}
		layers[i] = layer
	}

	reapCfg, err := LoadREAPConfig(filepath.Dir(path))
	if err != nil {
		return nil, fmt.Errorf("LoadGGUFLlama: REAP: %w", err)
	}
	if reapCfg == nil {
		reapCfg = InferREAPConfigFromName(filepath.Base(path))
	}

	m := &GGUFLlama{
		Config:       cfg,
		Layers:       layers,
		EmbedTokens:  embedTokens,
		OutputNorm:   outputNorm,
		LMHead:       lmHead,
		EmbedMatrix:  embedMatrix,
		LMHeadM:      lmHeadM,
		LMHeadGraph:  lmHeadGraph,
		UseGGMLQuant: useGGMLQuant,
		Backend:      backend,
		REAP:         reapCfg,
	}
	m.precomputeRoPE()
	if dg, dp, err := m.BuildDecodeGraph(); err == nil {
		m.DecodeGraph = dg
		m.DecodePlan = dp
	} else {
		return nil, fmt.Errorf("build decode graph: %w", err)
	}
	return m, nil
}

// ggufParseConfig extracts LLaMA/Qwen-style config from GGUF metadata.
func ggufParseConfig(g *gguf.GGUF) (GGUFLlamaConfig, error) {
	arch, _ := g.MetaString("general.architecture")
	if arch == "" {
		arch = "llama"
	}
	key := func(suffix string) string { return arch + "." + suffix }
	req := func(suffix string) (uint32, error) {
		if v, ok := g.MetaUint32(key(suffix)); ok {
			return v, nil
		}
		if arch != "llama" {
			if v, ok := g.MetaUint32("llama." + suffix); ok {
				return v, nil
			}
		}
		return 0, fmt.Errorf("missing metadata key %q", key(suffix))
	}
	hidden, err := req("embedding_length")
	if err != nil {
		return GGUFLlamaConfig{}, err
	}
	layers, err := req("block_count")
	if err != nil {
		return GGUFLlamaConfig{}, err
	}
	heads, err := req("attention.head_count")
	if err != nil {
		return GGUFLlamaConfig{}, err
	}
	kvHeads, err := req("attention.head_count_kv")
	if err != nil {
		return GGUFLlamaConfig{}, err
	}
	ffn, err := req("feed_forward_length")
	if err != nil {
		if v, ok := ggufMetaUint32Any(g, key("expert_feed_forward_length"), "llama.expert_feed_forward_length"); ok {
			ffn = v
		} else {
			return GGUFLlamaConfig{}, err
		}
	}
	maxCtx, err := req("context_length")
	if err != nil {
		return GGUFLlamaConfig{}, err
	}
	vocabSize, err := req("vocab_size")
	if err != nil {
		if t, ok := g.TensorByName("token_embd.weight"); ok && len(t.Shape) >= 2 {
			vocabSize = uint32(t.Shape[1])
		} else {
			// fallback for tiny synthetic fixtures without an embedding tensor
			vocabSize = 32000
		}
	}
	ropeBase := float32(10000.0)
	if v, ok := ggufMetaFloat32Any(g, key("rope.freq_base"), "llama.rope.freq_base"); ok {
		ropeBase = v
	}
	eps := float32(1e-5)
	if v, ok := ggufMetaFloat32Any(g, key("attention.layer_norm_rms_epsilon"), "llama.attention.layer_norm_rms_epsilon"); ok {
		eps = v
	}
	ropeDim := int(hidden) / int(heads)
	if v, ok := ggufMetaUint32Any(g, key("rope.dimension_count"), "llama.rope.dimension_count"); ok {
		ropeDim = int(v)
	}
	experts, _ := ggufMetaUint32Any(g, key("expert_count"), "llama.expert_count")
	expertsPerTok, _ := ggufMetaUint32Any(g, key("expert_used_count"), "llama.expert_used_count")
	moeHidden, _ := ggufMetaUint32Any(g, key("expert_feed_forward_length"), "llama.expert_feed_forward_length")
	sharedMoEHidden, _ := ggufMetaUint32Any(g, key("expert_shared_feed_forward_length"), "llama.expert_shared_feed_forward_length")
	fullAttnInterval, _ := ggufMetaUint32Any(g, key("full_attention_interval"), "llama.full_attention_interval")
	keyLen, _ := ggufMetaUint32Any(g, key("attention.key_length"), "llama.attention.key_length")
	valueLen, _ := ggufMetaUint32Any(g, key("attention.value_length"), "llama.attention.value_length")
	ssmConvKernel, _ := ggufMetaUint32Any(g, key("ssm.conv_kernel"), "llama.ssm.conv_kernel")
	ssmGroupCount, _ := ggufMetaUint32Any(g, key("ssm.group_count"), "llama.ssm.group_count")
	ssmInnerSize, _ := ggufMetaUint32Any(g, key("ssm.inner_size"), "llama.ssm.inner_size")
	ssmStateSize, _ := ggufMetaUint32Any(g, key("ssm.state_size"), "llama.ssm.state_size")
	ssmRank, _ := ggufMetaUint32Any(g, key("ssm.time_step_rank"), "llama.ssm.time_step_rank")
	bosID, _ := ggufMetaUint32Any(g, "tokenizer.ggml.bos_token_id")
	eosID, _ := ggufMetaUint32Any(g, "tokenizer.ggml.eos_token_id")
	cfg := GGUFLlamaConfig{
		Architecture:          arch,
		HiddenSize:            int(hidden),
		NumLayers:             int(layers),
		NumHeads:              int(heads),
		NumKVHeads:            int(kvHeads),
		HeadDim:               int(hidden) / int(heads),
		FFNHiddenSize:         int(ffn),
		VocabSize:             int(vocabSize),
		MaxSeqLen:             int(maxCtx),
		RMSNormEps:            eps,
		RopeFreqBase:          ropeBase,
		RopeDimCount:          ropeDim,
		NumExperts:            int(experts),
		NumExpertsPerTok:      int(expertsPerTok),
		MoEHiddenSize:         int(moeHidden),
		SharedMoEHiddenSize:   int(sharedMoEHidden),
		FullAttentionInterval: int(fullAttnInterval),
		AttentionKeyLength:    int(keyLen),
		AttentionValueLength:  int(valueLen),
		SSMConvKernel:         int(ssmConvKernel),
		SSMGroupCount:         int(ssmGroupCount),
		SSMInnerSize:          int(ssmInnerSize),
		SSMStateSize:          int(ssmStateSize),
		SSMTimeStepRank:       int(ssmRank),
		BOSTokenID:            int(bosID),
		EOSTokenID:            int(eosID),
	}
	if cfg.AttentionKeyLength > 0 {
		cfg.HeadDim = cfg.AttentionKeyLength
	}
	return cfg, nil
}

func loadGGUFMoEExpertMatrices(g *gguf.GGUF, layerIdx int) (router *gguf.QuantMatrix, gate, up, down *gguf.ExpertMatrices, err error) {
	loadMatrix := func(name string) (*gguf.QuantMatrix, error) {
		t, ok := g.TensorByName(name)
		if !ok {
			return nil, fmt.Errorf("tensor %q not found", name)
		}
		return g.MatrixFromTensor(t)
	}
	loadExperts := func(name string) (*gguf.ExpertMatrices, error) {
		t, ok := g.TensorByName(name)
		if !ok {
			return nil, fmt.Errorf("tensor %q not found", name)
		}
		return g.ExpertMatricesFromTensor(t)
	}
	p := fmt.Sprintf("blk.%d.", layerIdx)
	router, err = loadMatrix(p + "ffn_gate_inp.weight")
	if err != nil {
		return nil, nil, nil, nil, err
	}
	gate, err = loadExperts(p + "ffn_gate_exps.weight")
	if err != nil {
		return nil, nil, nil, nil, err
	}
	up, err = loadExperts(p + "ffn_up_exps.weight")
	if err != nil {
		return nil, nil, nil, nil, err
	}
	down, err = loadExperts(p + "ffn_down_exps.weight")
	if err != nil {
		return nil, nil, nil, nil, err
	}
	return router, gate, up, down, nil
}

func (c GGUFLlamaConfig) ValidateRuntimeSupported() error {
	if err := c.ValidateQwenNextHybridMetadata(); err != nil {
		return err
	}
	if c.NumExperts < 0 || c.NumExpertsPerTok < 0 || c.MoEHiddenSize < 0 {
		return fmt.Errorf("invalid GGUF MoE metadata experts=%d active=%d moe_hidden=%d", c.NumExperts, c.NumExpertsPerTok, c.MoEHiddenSize)
	}
	if c.NumExperts > 0 && (c.NumExpertsPerTok <= 0 || c.MoEHiddenSize <= 0) {
		return fmt.Errorf("incomplete GGUF MoE metadata for architecture %q (experts=%d active=%d moe_hidden=%d)", c.Architecture, c.NumExperts, c.NumExpertsPerTok, c.MoEHiddenSize)
	}
	return nil
}

func ggufMetaUint32Any(g *gguf.GGUF, keys ...string) (uint32, bool) {
	for _, key := range keys {
		if v, ok := g.MetaUint32(key); ok {
			return v, true
		}
	}
	return 0, false
}

func ggufMetaFloat32Any(g *gguf.GGUF, keys ...string) (float32, bool) {
	for _, key := range keys {
		if v, ok := g.MetaFloat32(key); ok {
			return v, true
		}
	}
	return 0, false
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

func (m *GGUFLlama) gemvMaybe(out, x, wf32 []float32, wq *gguf.QuantMatrix, inDim, outDim int) {
	if wq != nil {
		if m.UseGGMLQuant {
			if err := m.gemvGGMLQuant(out, x, wq, inDim, outDim); err == nil {
				return
			}
		}
		if err := gemvGGUFQuantRows(out, x, wq, inDim, outDim); err == nil {
			return
		}
	}
	if len(wf32) < inDim*outDim {
		for i := 0; i < outDim && i < len(out); i++ {
			out[i] = 0
		}
		return
	}
	m.gemv(out, x, wf32, inDim, outDim)
}

func gemvGGUFQuantRows(out, x []float32, w *gguf.QuantMatrix, inDim, outDim int) error {
	if w == nil || w.InDim != inDim || w.OutDim != outDim || len(out) < outDim || len(x) < inDim {
		return fmt.Errorf("bad quant matrix dims")
	}
	if err := gemvGGMLQuantRows(out, x, w, inDim, outDim); err == nil {
		return nil
	}
	return gemvGGUFQuantRowsPureGo(out, x, w, inDim, outDim)
}

func gemvGGUFQuantRowsPureGo(out, x []float32, w *gguf.QuantMatrix, inDim, outDim int) error {
	workers := runtime.GOMAXPROCS(0)
	if workers < 1 {
		workers = 1
	}
	if outDim < workers*8 {
		workers = 1
	}
	if workers == 1 {
		row := make([]float32, inDim)
		for r := 0; r < outDim; r++ {
			if err := w.DequantRowTo(row, r); err != nil {
				return err
			}
			out[r] = dotF32(row, x[:inDim])
		}
		return nil
	}
	var wg sync.WaitGroup
	errCh := make(chan error, workers)
	chunk := (outDim + workers - 1) / workers
	for wid := 0; wid < workers; wid++ {
		start := wid * chunk
		end := start + chunk
		if end > outDim {
			end = outDim
		}
		if start >= end {
			continue
		}
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			row := make([]float32, inDim)
			for r := start; r < end; r++ {
				if err := w.DequantRowTo(row, r); err != nil {
					errCh <- err
					return
				}
				out[r] = dotF32(row, x[:inDim])
			}
		}(start, end)
	}
	wg.Wait()
	select {
	case err := <-errCh:
		return err
	default:
		return nil
	}
}

func dotF32(a, b []float32) float32 {
	var sum float32
	for i, av := range a {
		sum += av * b[i]
	}
	return sum
}

func gemvGGMLQuantRows(out, x []float32, w *gguf.QuantMatrix, inDim, outDim int) error {
	if w == nil || len(out) < outDim || len(x) < inDim || w.InDim != inDim || w.OutDim != outDim || !ggmlquant.HasVecDot(int(w.QType)) {
		return fmt.Errorf("ggml quant rows unavailable")
	}
	vt := ggmlquant.VecDotType(int(w.QType))
	xRaw, err := ggmlquant.QuantizeFromFloat(vt, x[:inDim])
	if err != nil {
		return err
	}
	rowBytes, err := w.RowBytes()
	if err != nil {
		return err
	}
	return ggmlquant.VecDotRows(int(w.QType), out[:outDim], w.Raw, rowBytes, xRaw, inDim, outDim)
}

func (m *GGUFLlama) gemvGGMLQuant(out, x []float32, w *gguf.QuantMatrix, inDim, outDim int) error {
	if w.InDim != inDim || w.OutDim != outDim {
		return fmt.Errorf("bad quant matrix dims")
	}
	vt := ggmlquant.VecDotType(int(w.QType))
	xRaw, err := ggmlquant.QuantizeFromFloat(vt, x[:inDim])
	if err != nil {
		return err
	}
	rowBytes, err := w.RowBytes()
	if err != nil {
		return err
	}
	return ggmlquant.VecDotRows(int(w.QType), out[:outDim], w.Raw, rowBytes, xRaw, inDim, outDim)
}

func (m *GGUFLlama) gemvGGMLQuantRaw(out []float32, xRaw []byte, w *gguf.QuantMatrix, inDim, outDim int) error {
	if w == nil || w.InDim != inDim || w.OutDim != outDim {
		return fmt.Errorf("bad quant matrix dims")
	}
	rowBytes, err := w.RowBytes()
	if err != nil {
		return err
	}
	return ggmlquant.VecDotRows(int(w.QType), out[:outDim], w.Raw, rowBytes, xRaw, inDim, outDim)
}

func quantActQ8K(x []float32) ([]byte, error) {
	return ggmlquant.QuantizeFromFloat(ggmlquant.Q8_K, x)
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

// GGUFForwardState holds reusable per-token scratch buffers for GGUFLlama.ForwardState.
// Keeping this around avoids allocating ~hundreds of KB of temporary buffers per token.
type GGUFForwardState struct {
	hidden       []float32
	attnIn       []float32
	q            []float32
	k            []float32
	v            []float32
	attnOut      []float32
	attnScores   []float32
	oOut         []float32
	ffnIn        []float32
	gate         []float32
	up           []float32
	ffnMid       []float32
	down         []float32
	logits       []float32
	qwenNext     []GGUFQwenNextState
	compressedKV []*kv.CompressedKVCache
}

// NewForwardState allocates reusable scratch for one autoregressive stream.
func (m *GGUFLlama) NewForwardState() *GGUFForwardState {
	cfg := m.Config
	h := cfg.HiddenSize
	nH := cfg.NumHeads
	hDim := cfg.HeadDim
	kvDim := cfg.NumKVHeads * hDim
	ffn := cfg.FFNHiddenSize
	if cfg.MoEHiddenSize > ffn {
		ffn = cfg.MoEHiddenSize
	}
	qwenNext := make([]GGUFQwenNextState, cfg.NumLayers)
	if cfg.IsQwenNextHybridGGUF() {
		for i := range qwenNext {
			qwenNext[i], _ = cfg.NewQwenNextState()
		}
	}
	return &GGUFForwardState{
		hidden:     make([]float32, h),
		attnIn:     make([]float32, h),
		q:          make([]float32, nH*hDim),
		k:          make([]float32, kvDim),
		v:          make([]float32, kvDim),
		attnOut:    make([]float32, nH*hDim),
		attnScores: make([]float32, cfg.MaxSeqLen),
		oOut:       make([]float32, h),
		ffnIn:      make([]float32, h),
		gate:       make([]float32, ffn),
		up:         make([]float32, ffn),
		ffnMid:     make([]float32, ffn),
		down:       make([]float32, h),
		logits:     make([]float32, cfg.VocabSize),
		qwenNext:   qwenNext,
	}
}

// Forward runs a single token through the model, updating the KV cache,
// and returns the logits vector [vocabSize].
//
// kvK[layer][step*kvDim : (step+1)*kvDim] and kvV[...] are the KV caches.
func (m *GGUFLlama) Forward(tokenID, step int, kvK, kvV [][]float32) []float32 {
	return m.ForwardState(m.NewForwardState(), tokenID, step, kvK, kvV)
}

// ForwardState is Forward using caller-owned reusable scratch buffers.
func (m *GGUFLlama) ForwardState(st *GGUFForwardState, tokenID, step int, kvK, kvV [][]float32) []float32 {
	cfg := m.Config
	h := cfg.HiddenSize
	nH := cfg.NumHeads
	nKV := cfg.NumKVHeads
	hDim := cfg.HeadDim
	kvDim := nKV * hDim
	ffn := cfg.FFNHiddenSize
	if cfg.MoEHiddenSize > ffn {
		ffn = cfg.MoEHiddenSize
	}
	rotHalf := m.rotHalf

	// Token embedding
	hidden := st.hidden[:h]
	if m.UseGGMLQuant && m.EmbedMatrix != nil {
		if err := m.EmbedMatrix.DequantRowTo(hidden, tokenID); err != nil {
			return st.logits[:cfg.VocabSize]
		}
	} else {
		copy(hidden, m.EmbedTokens[tokenID*h:(tokenID+1)*h])
	}

	// RoPE frequencies for this position
	posFreqs := m.ropeFreqs[step*rotHalf : (step+1)*rotHalf]

	attnIn := st.attnIn[:h]
	q := st.q[:nH*hDim]
	k := st.k[:kvDim]
	v := st.v[:kvDim]
	attnOut := st.attnOut[:nH*hDim]
	attnScores := st.attnScores[:cfg.MaxSeqLen]
	oOut := st.oOut[:h]
	ffnIn := st.ffnIn[:h]
	gate := st.gate[:ffn]
	up := st.up[:ffn]
	ffnMid := st.ffnMid[:ffn]
	down := st.down[:h]

	for i, layer := range m.Layers {
		// ── attention / QwenNext hybrid sub-layer ─────────────────────────
		copy(attnIn, hidden)
		m.rmsNorm(attnIn, layer.AttnNorm)
		hybridDone := false
		if layer.HasQwenNextHybridTensors() && i < len(st.qwenNext) {
			if err := m.forwardQwenNextHybridLayer(oOut, &layer, &st.qwenNext[i], attnIn); err == nil {
				for j := range hidden {
					hidden[j] += oOut[j]
				}
				hybridDone = true
			}
		}
		if !hybridDone {
			qOut := nH * hDim
			if layer.WQm != nil && layer.WQm.OutDim == qOut*2 {
				qFull := make([]float32, qOut*2)
				m.gemvMaybe(qFull, attnIn, layer.WQ, layer.WQm, h, qOut*2)
				copyGGUFQwenFullQ(q, qFull, nH, hDim)
			} else if m.UseGGMLQuant && layer.WQm != nil {
				if actRaw, err := quantActQ8K(attnIn[:h]); err == nil {
					_ = m.gemvGGMLQuantRaw(q, actRaw, layer.WQm, h, qOut)
				} else {
					m.gemvMaybe(q, attnIn, layer.WQ, layer.WQm, h, qOut)
				}
			} else {
				m.gemvMaybe(q, attnIn, layer.WQ, layer.WQm, h, qOut)
			}
			m.gemvMaybe(k, attnIn, layer.WK, layer.WKm, h, kvDim)
			m.gemvMaybe(v, attnIn, layer.WV, layer.WVm, h, kvDim)
			if len(layer.QNorm) == hDim {
				normGGUFHeads(q, layer.QNorm, nH, hDim, cfg.RMSNormEps)
			}
			if len(layer.KNorm) == hDim {
				normGGUFHeads(k, layer.KNorm, nKV, hDim, cfg.RMSNormEps)
			}

			// RoPE
			applyRoPEInplace(q, posFreqs, nH, hDim, rotHalf)
			applyRoPEInplace(k, posFreqs, nKV, hDim, rotHalf)

			// Update KV cache
			var kCache, vCache []float32
			seqLen := step + 1
			if i < len(st.compressedKV) && st.compressedKV[i] != nil {
				st.compressedKV[i].Append(k, v)
				kCache = st.compressedKV[i].GetK()
				vCache = st.compressedKV[i].GetV()
				seqLen = st.compressedKV[i].SeqLen()
			} else {
				kCache = kvK[i]
				vCache = kvV[i]
				copy(kCache[step*kvDim:], k)
				copy(vCache[step*kvDim:], v)
			}

			// Grouped-query attention: compute attention output
			m.gqaAttentionInto(attnOut, attnScores, q, kCache, vCache, seqLen, nH, nKV, hDim)

			// Output projection
			m.gemvMaybe(oOut, attnOut, layer.WO, layer.WOm, nH*hDim, h)

			// Residual add
			for j := range hidden {
				hidden[j] += oOut[j]
			}
		}

		// ── FFN sub-layer ─────────────────────────────────────────────────
		copy(ffnIn, hidden)
		m.rmsNorm(ffnIn, layer.FFNNorm)

		if layer.ExpertGateM != nil && cfg.NumExperts > 0 {
			m.ggufMoEForward(down, gate, up, ffnMid, ffnIn, &layer, i)
		} else {
			if m.UseGGMLQuant && layer.WGateM != nil {
				if ffnRaw, err := quantActQ8K(ffnIn[:h]); err == nil {
					_ = m.gemvGGMLQuantRaw(gate, ffnRaw, layer.WGateM, h, ffn)
					_ = m.gemvGGMLQuantRaw(up, ffnRaw, layer.WUpM, h, ffn)
				} else {
					m.gemvMaybe(gate, ffnIn, layer.WGate, layer.WGateM, h, ffn)
					m.gemvMaybe(up, ffnIn, layer.WUp, layer.WUpM, h, ffn)
				}
			} else {
				m.gemvMaybe(gate, ffnIn, layer.WGate, layer.WGateM, h, ffn)
				m.gemvMaybe(up, ffnIn, layer.WUp, layer.WUpM, h, ffn)
			}

			m.siluMul(ffnMid, gate, up)

			m.gemvMaybe(down, ffnMid, layer.WDown, layer.WDownM, ffn, h)
		}

		for j := range hidden {
			hidden[j] += down[j]
		}
	}

	// Final norm + LM head
	m.rmsNorm(hidden, m.OutputNorm)
	logits := st.logits[:cfg.VocabSize]
	if m.LMHeadGraph != nil {
		_ = m.LMHeadGraph.Run(hidden, logits)
	} else {
		m.gemvMaybe(logits, hidden, m.LMHead, m.LMHeadM, h, cfg.VocabSize)
	}
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
type GGUFGenerationOptions struct {
	CacheTypeK       string
	CacheTypeV       string
	KVResidualWindow int
}

func (m *GGUFLlama) Generate(promptIDs []int, maxNew int) ([]int, error) {
	return m.GenerateWithOptions(promptIDs, maxNew, GGUFGenerationOptions{KVResidualWindow: -1})
}

func (m *GGUFLlama) GenerateWithOptions(promptIDs []int, maxNew int, opts GGUFGenerationOptions) ([]int, error) {
	cfg := m.Config
	kvDim := cfg.NumKVHeads * cfg.HeadDim
	maxSeq := len(promptIDs) + maxNew
	if maxSeq > cfg.MaxSeqLen {
		maxSeq = cfg.MaxSeqLen
	}

	// Allocate KV caches. TurboQuant options are accepted and validated here;
	// QwenNext hybrid layers use recurrent state and full-attention interval
	// layers still consume F32 KV slices until attention reads compressed caches directly.
	var compressedKV []*kv.CompressedKVCache
	if opts.CacheTypeK != "" || opts.CacheTypeV != "" || opts.KVResidualWindow >= 0 {
		caches, err := m.NewTurboQuantKVCache(opts.CacheTypeK, opts.CacheTypeV, opts.KVResidualWindow)
		if err != nil {
			return nil, err
		}
		compressedKV = caches
	}
	kvK := make([][]float32, cfg.NumLayers)
	kvV := make([][]float32, cfg.NumLayers)
	for i := range kvK {
		kvK[i] = make([]float32, maxSeq*kvDim)
		kvV[i] = make([]float32, maxSeq*kvDim)
	}

	var generated []int
	state := m.NewForwardState()
	state.compressedKV = compressedKV
	if len(promptIDs) == 0 || maxNew <= 0 {
		return generated, nil
	}

	// Prefill: run all prompt tokens once and keep logits from the final prompt token.
	var logits []float32
	for step, tok := range promptIDs {
		logits = m.ForwardState(state, tok, step, kvK, kvV)
	}

	// Decode: autoregressively generate maxNew tokens.
	step := len(promptIDs) - 1
	for range maxNew {
		next := argmaxF32(logits)
		generated = append(generated, next)
		if cfg.IsEOS(next) {
			break
		}
		step++
		if step >= maxSeq {
			break
		}
		logits = m.ForwardState(state, next, step, kvK, kvV)
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

func copyGGUFQwenFullQ(dst, qFull []float32, heads, headDim int) {
	for h := 0; h < heads; h++ {
		src := h * headDim * 2
		copy(dst[h*headDim:(h+1)*headDim], qFull[src:src+headDim])
	}
}

func normGGUFHeads(x, weight []float32, heads, headDim int, eps float32) {
	for h := 0; h < heads; h++ {
		row := x[h*headDim : (h+1)*headDim]
		var ss float32
		for _, v := range row {
			ss += v * v
		}
		scale := float32(1.0 / math.Sqrt(float64(ss/float32(headDim)+eps)))
		for i := 0; i < headDim; i++ {
			row[i] *= scale * weight[i]
		}
	}
}

func (c GGUFLlamaConfig) IsEOS(tokenID int) bool {
	if tokenID < 0 {
		return false
	}
	if c.EOSTokenID > 0 {
		return tokenID == c.EOSTokenID
	}
	return tokenID == c.VocabSize-1 || (c.VocabSize > 2 && tokenID == 2)
}

// SetCompressedKVForSmoke attaches native compressed KV caches to a reusable
// forward state. It is primarily used by smoke/benchmark tooling; generation
// APIs set this internally when TurboQuant options are supplied.
func (st *GGUFForwardState) SetCompressedKVForSmoke(caches []*kv.CompressedKVCache) {
	if st != nil {
		st.compressedKV = caches
	}
}
