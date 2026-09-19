package model

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"sync"
	"time"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	"github.com/rcarmo/go-pherence/internal/checked"
	loaderconfig "github.com/rcarmo/go-pherence/loader/config"
	"github.com/rcarmo/go-pherence/loader/weights"
)

// FrozenGPUOptions configures the experimental compact GPU encoder.
type FrozenGPUOptions struct {
	MaxTokens    int
	BudgetBytes  uint64
	ReserveBytes uint64
}

// FrozenGPUStats reports the encoder's static load-time footprint.
type FrozenGPUStats struct {
	ModelDir         string
	MaxTokens        int
	BudgetBytes      uint64
	ReserveBytes     uint64
	FreeBytesAtLoad  uint64
	TotalBytesAtLoad uint64
	EstimatedBytes   uint64
	AllocatedBytes   uint64
	MatrixBytes      uint64
	NormBytes        uint64
	KVBytes          uint64
	RopeBytes        uint64
	ScratchBytes     uint64
	LoadDuration     time.Duration
}

type frozenGPUHostLayer struct {
	inputNorm []float32
	postNorm  []float32
	qNorm     []float32
	kNorm     []float32

	qProj    []byte
	kProj    []byte
	vProj    []byte
	oProj    []byte
	gateProj []byte
	upProj   []byte
	downProj []byte

	headDim int
	qDim    int
	kvDim   int
}

type frozenGPUHostModel struct {
	embedRaw      []byte
	finalNorm     []float32
	layers        []frozenGPUHostLayer
	maxMatrixElem int
	matrixBytes   uint64
	normBytes     uint64
}

type frozenGPULayer struct {
	inputNorm *nvidia.Buffer
	postNorm  *nvidia.Buffer
	qNorm     *nvidia.Buffer
	kNorm     *nvidia.Buffer

	qProj    *nvidia.Buffer
	kProj    *nvidia.Buffer
	vProj    *nvidia.Buffer
	oProj    *nvidia.Buffer
	gateProj *nvidia.Buffer
	upProj   *nvidia.Buffer
	downProj *nvidia.Buffer

	headDim int
	qDim    int
	kvDim   int
}

func (l *frozenGPULayer) free() {
	if l == nil {
		return
	}
	for _, buf := range []*nvidia.Buffer{l.inputNorm, l.postNorm, l.qNorm, l.kNorm, l.qProj, l.kProj, l.vProj, l.oProj, l.gateProj, l.upProj, l.downProj} {
		if buf != nil {
			buf.Free()
		}
	}
}

// FrozenGPUEncoder owns a compact BF16 Qwen3 encoder with GPU-resident state.
type FrozenGPUEncoder struct {
	mu     sync.Mutex
	closed bool

	dir       string
	cfg       LlamaConfig
	maxTokens int
	source    weights.Source
	stats     FrozenGPUStats

	embedRaw []byte
	embedRow []float32

	layers    []frozenGPULayer
	finalNorm *nvidia.Buffer
	ropeTable *nvidia.DevBuf

	weightScratch *nvidia.Buffer

	hidden   *nvidia.DevBuf
	residual *nvidia.DevBuf
	normed   *nvidia.DevBuf
	q        *nvidia.DevBuf
	qNormed  *nvidia.DevBuf
	k        *nvidia.DevBuf
	kNormed  *nvidia.DevBuf
	v        *nvidia.DevBuf
	attnOut  *nvidia.DevBuf
	oOut     *nvidia.DevBuf
	gate     *nvidia.DevBuf
	up       *nvidia.DevBuf
	down     *nvidia.DevBuf
}

// NewFrozenGPUEncoder loads an experimental compact BF16-only dense Qwen3
// checkpoint without materializing full-F32 transformer matrices.
func NewFrozenGPUEncoder(dir string, options FrozenGPUOptions) (enc *FrozenGPUEncoder, err error) {
	start := time.Now()
	if strings.TrimSpace(dir) == "" {
		return nil, fmt.Errorf("empty model directory")
	}
	if options.BudgetBytes == 0 || options.ReserveBytes == 0 {
		return nil, fmt.Errorf("explicit positive GPU budget and reserve required")
	}
	cfg, err := readFrozenGPUConfig(dir)
	if err != nil {
		return nil, err
	}
	maxTokens, err := frozenGPUResolveMaxTokens(cfg, options)
	if err != nil {
		return nil, err
	}
	if err := validateFrozenGPUConfig(cfg, maxTokens); err != nil {
		return nil, err
	}
	if !nvidia.SgemmReady() {
		return nil, fmt.Errorf("GPU not available")
	}

	source, err := weights.OpenSafetensors(dir)
	if err != nil {
		return nil, err
	}
	enc = &FrozenGPUEncoder{
		dir:       dir,
		cfg:       cfg,
		maxTokens: maxTokens,
		source:    source,
		stats: FrozenGPUStats{
			ModelDir:     dir,
			MaxTokens:    maxTokens,
			BudgetBytes:  options.BudgetBytes,
			ReserveBytes: options.ReserveBytes,
		},
	}
	owned := enc // named return enc may be overwritten by return nil, err
	defer func() {
		if err != nil {
			owned.Close()
			enc = nil
		}
	}()

	host, err := loadFrozenGPUHostModel(source, cfg)
	if err != nil {
		return nil, err
	}
	bytesPlan, err := estimateFrozenGPUBytes(cfg, host, maxTokens)
	if err != nil {
		return nil, err
	}
	freeBytes, totalBytes := nvidia.MemInfo()
	enc.stats.FreeBytesAtLoad = freeBytes
	enc.stats.TotalBytesAtLoad = totalBytes
	enc.stats.EstimatedBytes = bytesPlan.total
	enc.stats.MatrixBytes = bytesPlan.matrix
	enc.stats.NormBytes = bytesPlan.norm
	enc.stats.KVBytes = bytesPlan.kv
	enc.stats.RopeBytes = bytesPlan.rope
	enc.stats.ScratchBytes = bytesPlan.scratch
	if err := checkFrozenGPUBudget(bytesPlan.total, freeBytes, options); err != nil {
		return nil, err
	}

	if err := enc.allocateFrozenGPU(host, bytesPlan); err != nil {
		return nil, err
	}
	enc.stats.LoadDuration = time.Since(start)
	return enc, nil
}

// EncodeTokenHiddenStates runs one causal prompt on the compact GPU encoder and
// returns one final-normalized hidden row per token.
func (e *FrozenGPUEncoder) EncodeTokenHiddenStates(ids []int) ([][]float32, error) {
	if e == nil {
		return nil, fmt.Errorf("nil frozen GPU encoder")
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.encodeTokenHiddenStatesLocked(ids, true, nil)
}

func (e *FrozenGPUEncoder) encodeTokenHiddenStatesLocked(ids []int, all bool, timing *FrozenGPUDecisionStats) ([][]float32, error) {
	if e.closed {
		return nil, fmt.Errorf("frozen GPU encoder is closed")
	}
	if err := validateFrozenGPURequest(e.cfg, e.maxTokens, ids); err != nil {
		return nil, err
	}
	batch, h := len(ids), e.cfg.HiddenSize
	started := time.Now()
	for pos, tok := range ids {
		if err := e.uploadEmbedding(tok, pos); err != nil {
			return nil, fmt.Errorf("token %d embedding: %w", pos, err)
		}
	}
	if timing != nil {
		if err := nvidia.SyncErr(); err != nil {
			return nil, err
		}
		timing.EmbeddingUploadSeconds = time.Since(started).Seconds()
		timing.UploadBytes = batch * h * 4
		started = time.Now()
	}
	for layerIdx := range e.layers {
		if err := e.forwardLayer(layerIdx, batch); err != nil {
			return nil, fmt.Errorf("layer %d: %w", layerIdx, err)
		}
	}
	if err := e.normRows(e.normed, e.hidden, e.finalNorm, batch, h); err != nil {
		return nil, err
	}
	if err := nvidia.SyncErr(); err != nil {
		return nil, err
	}
	if timing != nil {
		timing.PrefillSeconds = time.Since(started).Seconds()
		started = time.Now()
	}
	if !all {
		last := make([]float32, h)
		if err := e.normed.Slice((batch-1)*h, h).GPUBuffer().Download(last); err != nil {
			return nil, err
		}
		if timing != nil {
			timing.DownloadSeconds = time.Since(started).Seconds()
			timing.DownloadBytes = h * 4
		}
		return [][]float32{last}, nil
	}
	flat := make([]float32, batch*h)
	if err := e.normed.GPUBuffer().Download(flat); err != nil {
		return nil, err
	}
	rows := make([][]float32, batch)
	for pos := range rows {
		rows[pos] = flat[pos*h : (pos+1)*h]
	}
	return rows, nil
}

// Stats reports the encoder's load-time footprint.
func (e *FrozenGPUEncoder) Stats() FrozenGPUStats {
	if e == nil {
		return FrozenGPUStats{}
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.stats
}

// Close releases resources owned by the encoder. It is idempotent.
func (e *FrozenGPUEncoder) Close() {
	if e == nil {
		return
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return
	}
	e.closed = true
	// Buffers may still be referenced by queued kernels on an error path.
	nvidia.SyncAll()
	if e.source != nil {
		_ = e.source.Close()
		e.source = nil
	}
	for i := range e.layers {
		e.layers[i].free()
	}
	e.layers = nil
	for _, buf := range []*nvidia.Buffer{e.finalNorm, e.weightScratch} {
		if buf != nil {
			buf.Free()
		}
	}
	for _, buf := range []*nvidia.DevBuf{e.ropeTable, e.hidden, e.residual, e.normed, e.q, e.qNormed, e.k, e.kNormed, e.v, e.attnOut, e.oOut, e.gate, e.up, e.down} {
		if buf != nil {
			buf.Free()
		}
	}
	e.embedRaw = nil
	e.embedRow = nil
}

type frozenGPUBytePlan struct {
	total   uint64
	matrix  uint64
	norm    uint64
	kv      uint64
	rope    uint64
	scratch uint64
}

func readFrozenGPUConfig(dir string) (LlamaConfig, error) {
	var cfg LlamaConfig
	cfgData, err := loaderconfig.ReadModelConfig(dir, &cfg)
	if err != nil {
		return LlamaConfig{}, err
	}
	var policy struct {
		AttentionBias bool            `json:"attention_bias"`
		UseSliding    bool            `json:"use_sliding_window"`
		RopeScaling   json.RawMessage `json:"rope_scaling"`
		Quantization  json.RawMessage `json:"quantization_config"`
	}
	if err := json.Unmarshal(cfgData, &policy); err != nil {
		return LlamaConfig{}, err
	}
	if policy.AttentionBias || policy.UseSliding || (len(policy.RopeScaling) > 0 && string(policy.RopeScaling) != "null") || (len(policy.Quantization) > 0 && string(policy.Quantization) != "null") {
		return LlamaConfig{}, fmt.Errorf("compact encoder rejects attention bias, sliding, scaled RoPE or quantized config")
	}
	// This narrow encoder accepts only root dense Qwen3, not multimodal
	// wrappers whose nested config could bypass the policy checks above.
	if cfg.HiddenSize == 0 {
		return LlamaConfig{}, fmt.Errorf("root dense Qwen3 config required")
	}
	if cfg.HeadDim == 0 && cfg.NumHeads > 0 && cfg.HiddenSize%cfg.NumHeads == 0 {
		cfg.HeadDim = cfg.HiddenSize / cfg.NumHeads
	}
	return cfg, nil
}

func frozenGPUResolveMaxTokens(cfg LlamaConfig, options FrozenGPUOptions) (int, error) {
	maxTokens := options.MaxTokens
	if maxTokens < 0 {
		return 0, fmt.Errorf("negative max tokens")
	}
	if maxTokens == 0 {
		maxTokens = cfg.MaxSeqLen
		if maxTokens <= 0 || maxTokens > 512 {
			maxTokens = 512
		}
	}
	if maxTokens <= 0 {
		return 0, fmt.Errorf("invalid max tokens %d", maxTokens)
	}
	if maxTokens > 512 {
		return 0, fmt.Errorf("max tokens %d exceeds experimental limit 512", maxTokens)
	}
	if cfg.MaxSeqLen > 0 && maxTokens > cfg.MaxSeqLen {
		return 0, fmt.Errorf("max tokens %d exceeds model context %d", maxTokens, cfg.MaxSeqLen)
	}
	return maxTokens, nil
}

func validateFrozenGPUConfig(cfg LlamaConfig, maxTokens int) error {
	if cfg.ModelType != "qwen3" {
		return fmt.Errorf("unsupported model_type %q: only dense Qwen3 checkpoints are supported", cfg.ModelType)
	}
	if cfg.VocabSize <= 0 || cfg.HiddenSize <= 0 || cfg.Intermediate <= 0 || cfg.NumLayers <= 0 || cfg.NumHeads <= 0 || cfg.NumKVHeads <= 0 || cfg.HeadDim <= 0 {
		return fmt.Errorf("invalid Qwen3 config hidden=%d intermediate=%d layers=%d heads=%d kv_heads=%d head_dim=%d vocab=%d", cfg.HiddenSize, cfg.Intermediate, cfg.NumLayers, cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim, cfg.VocabSize)
	}
	if cfg.NumLayers > 128 || cfg.HiddenSize > 16384 || cfg.Intermediate > 65536 || cfg.NumHeads > 128 || cfg.HeadDim > 256 || cfg.HeadDim%2 != 0 || cfg.VocabSize > 1<<20 {
		return fmt.Errorf("config exceeds bounded dense Qwen3 limits")
	}
	if cfg.RopeTheta <= 0 || math.IsNaN(cfg.RopeTheta) || math.IsInf(cfg.RopeTheta, 0) {
		return fmt.Errorf("invalid rope theta")
	}
	if cfg.RMSNormEps <= 0 || math.IsNaN(cfg.RMSNormEps) || math.IsInf(cfg.RMSNormEps, 0) {
		return fmt.Errorf("invalid rms_norm_eps %g", cfg.RMSNormEps)
	}
	if cfg.NumHeads%cfg.NumKVHeads != 0 {
		return fmt.Errorf("unsupported grouped-query config heads=%d kv_heads=%d", cfg.NumHeads, cfg.NumKVHeads)
	}
	if cfg.HiddenAct != "" && cfg.HiddenAct != "silu" {
		return fmt.Errorf("unsupported hidden activation %q", cfg.HiddenAct)
	}
	if cfg.NumExperts > 0 || cfg.MoEIntermediate > 0 {
		return fmt.Errorf("unsupported mixture-of-experts Qwen3 config experts=%d moe_intermediate=%d", cfg.NumExperts, cfg.MoEIntermediate)
	}
	if cfg.HiddenPerLayer > 0 || cfg.VocabPerLayer > 0 || cfg.NumKVSharedLayers > 0 {
		return fmt.Errorf("unsupported per-layer/shared-KV config hidden_per_layer=%d vocab_per_layer=%d shared_kv_layers=%d", cfg.HiddenPerLayer, cfg.VocabPerLayer, cfg.NumKVSharedLayers)
	}
	if cfg.SlidingWindow > 0 || cfg.SlidingWindowPattern > 0 {
		return fmt.Errorf("unsupported sliding-window attention config window=%d pattern=%d", cfg.SlidingWindow, cfg.SlidingWindowPattern)
	}
	for i, lt := range cfg.LayerTypes {
		if lt != "" && lt != "full_attention" {
			return fmt.Errorf("unsupported layer_types[%d]=%q", i, lt)
		}
	}
	if cfg.AttentionKEqV {
		return fmt.Errorf("unsupported attention_k_eq_v=true")
	}
	if cfg.AttentionLogitSoftcapping != 0 || cfg.FinalLogitSoftcapping != 0 {
		return fmt.Errorf("unsupported Qwen3 softcapping attn=%g final=%g", cfg.AttentionLogitSoftcapping, cfg.FinalLogitSoftcapping)
	}
	if maxTokens <= 0 || maxTokens > 512 {
		return fmt.Errorf("invalid max token bound %d", maxTokens)
	}
	qDim, okQ := checked.MulInt(cfg.NumHeads, cfg.HeadDim)
	kvDim, okKV := checked.MulInt(cfg.NumKVHeads, cfg.HeadDim)
	if !okQ || !okKV || qDim <= 0 || kvDim <= 0 {
		return fmt.Errorf("projection dimension overflow q=%d*%d kv=%d*%d", cfg.NumHeads, cfg.HeadDim, cfg.NumKVHeads, cfg.HeadDim)
	}
	if _, ok := checked.MulInt(cfg.NumLayers, kvDim); !ok {
		return fmt.Errorf("KV dimension overflow layers=%d kv_dim=%d", cfg.NumLayers, kvDim)
	}
	return nil
}

func validateFrozenGPURequest(cfg LlamaConfig, maxTokens int, ids []int) error {
	if len(ids) == 0 {
		return fmt.Errorf("empty token sequence")
	}
	if len(ids) > maxTokens {
		return fmt.Errorf("token sequence len=%d exceeds encoder max tokens %d", len(ids), maxTokens)
	}
	for i, id := range ids {
		if id < 0 || id >= cfg.VocabSize {
			return fmt.Errorf("token %d id %d out of range [0,%d)", i, id, cfg.VocabSize)
		}
	}
	return nil
}

func loadFrozenGPUHostModel(source weights.Source, cfg LlamaConfig) (*frozenGPUHostModel, error) {
	host := &frozenGPUHostModel{layers: make([]frozenGPUHostLayer, cfg.NumLayers)}
	h := cfg.HiddenSize
	qDim, ok := checked.MulInt(cfg.NumHeads, cfg.HeadDim)
	if !ok {
		return nil, fmt.Errorf("Q dimension overflow")
	}
	kvDim, ok := checked.MulInt(cfg.NumKVHeads, cfg.HeadDim)
	if !ok {
		return nil, fmt.Errorf("KV dimension overflow")
	}

	embedRaw, dtype, shape, err := source.GetRaw("model.embed_tokens.weight")
	if err != nil {
		return nil, fmt.Errorf("load model.embed_tokens.weight: %w", err)
	}
	if dtype != "BF16" {
		return nil, fmt.Errorf("model.embed_tokens.weight dtype=%s, want BF16", dtype)
	}
	if err := frozenGPUExpectShape("model.embed_tokens.weight", shape, cfg.VocabSize, h); err != nil {
		return nil, err
	}
	embedElements, ok := checked.MulInt(cfg.VocabSize, h)
	if !ok || embedElements > int(^uint(0)>>1)/2 || len(embedRaw) != embedElements*2 {
		return nil, fmt.Errorf("embedding byte length mismatch")
	}
	host.embedRaw = embedRaw

	host.finalNorm, err = loadFrozenGPUFloatVector(source, "model.norm.weight", h)
	if err != nil {
		return nil, err
	}
	normBytes, err := frozenGPUVectorBytes(h)
	if err != nil {
		return nil, err
	}
	if host.normBytes, err = frozenGPUAddBytes(host.normBytes, normBytes); err != nil {
		return nil, err
	}

	for i := 0; i < cfg.NumLayers; i++ {
		prefix := fmt.Sprintf("model.layers.%d", i)
		if err := rejectFrozenGPUTensorIfPresent(source, prefix+".self_attn.q_proj.bias"); err != nil {
			return nil, err
		}
		if err := rejectFrozenGPUTensorIfPresent(source, prefix+".self_attn.k_proj.bias"); err != nil {
			return nil, err
		}
		if err := rejectFrozenGPUTensorIfPresent(source, prefix+".self_attn.v_proj.bias"); err != nil {
			return nil, err
		}
		layer := &host.layers[i]
		layer.headDim = cfg.HeadDim
		layer.qDim = qDim
		layer.kvDim = kvDim
		layer.inputNorm, err = loadFrozenGPUFloatVector(source, prefix+".input_layernorm.weight", h)
		if err != nil {
			return nil, err
		}
		layer.postNorm, err = loadFrozenGPUFloatVector(source, prefix+".post_attention_layernorm.weight", h)
		if err != nil {
			return nil, err
		}
		layer.qNorm, err = loadFrozenGPUFloatVector(source, prefix+".self_attn.q_norm.weight", cfg.HeadDim)
		if err != nil {
			return nil, err
		}
		layer.kNorm, err = loadFrozenGPUFloatVector(source, prefix+".self_attn.k_norm.weight", cfg.HeadDim)
		if err != nil {
			return nil, err
		}
		layer.qProj, err = loadFrozenGPUBF16Matrix(source, prefix+".self_attn.q_proj.weight", qDim, h)
		if err != nil {
			return nil, err
		}
		layer.kProj, err = loadFrozenGPUBF16Matrix(source, prefix+".self_attn.k_proj.weight", kvDim, h)
		if err != nil {
			return nil, err
		}
		layer.vProj, err = loadFrozenGPUBF16Matrix(source, prefix+".self_attn.v_proj.weight", kvDim, h)
		if err != nil {
			return nil, err
		}
		layer.oProj, err = loadFrozenGPUBF16Matrix(source, prefix+".self_attn.o_proj.weight", h, qDim)
		if err != nil {
			return nil, err
		}
		layer.gateProj, err = loadFrozenGPUBF16Matrix(source, prefix+".mlp.gate_proj.weight", cfg.Intermediate, h)
		if err != nil {
			return nil, err
		}
		layer.upProj, err = loadFrozenGPUBF16Matrix(source, prefix+".mlp.up_proj.weight", cfg.Intermediate, h)
		if err != nil {
			return nil, err
		}
		layer.downProj, err = loadFrozenGPUBF16Matrix(source, prefix+".mlp.down_proj.weight", h, cfg.Intermediate)
		if err != nil {
			return nil, err
		}
		for _, n := range []int{h, h, cfg.HeadDim, cfg.HeadDim} {
			normBytes, err := frozenGPUVectorBytes(n)
			if err != nil {
				return nil, err
			}
			host.normBytes, err = frozenGPUAddBytes(host.normBytes, normBytes)
			if err != nil {
				return nil, err
			}
		}
		for _, raw := range [][]byte{layer.qProj, layer.kProj, layer.vProj, layer.oProj, layer.gateProj, layer.upProj, layer.downProj} {
			bytes, err := frozenGPUPaddedRawBytes(raw)
			if err != nil {
				return nil, err
			}
			host.matrixBytes, err = frozenGPUAddBytes(host.matrixBytes, bytes)
			if err != nil {
				return nil, err
			}
			elems := len(raw) / 2
			if elems > host.maxMatrixElem {
				host.maxMatrixElem = elems
			}
		}
	}
	return host, nil
}

func loadFrozenGPUFloatVector(source weights.Source, name string, n int) ([]float32, error) {
	data, shape, err := source.GetFloat32(name)
	if err != nil {
		return nil, fmt.Errorf("load %s: %w", name, err)
	}
	if err := frozenGPUExpectVector(name, shape, n); err != nil {
		return nil, err
	}
	if len(data) != n {
		return nil, fmt.Errorf("%s len=%d want %d", name, len(data), n)
	}
	for _, v := range data {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return nil, fmt.Errorf("nonfinite norm %s", name)
		}
	}
	return data, nil
}

func loadFrozenGPUBF16Matrix(source weights.Source, name string, rows, cols int) ([]byte, error) {
	raw, dtype, shape, err := source.GetRaw(name)
	if err != nil {
		return nil, fmt.Errorf("load %s: %w", name, err)
	}
	if dtype != "BF16" {
		return nil, fmt.Errorf("%s dtype=%s, want BF16", name, dtype)
	}
	if err := frozenGPUExpectShape(name, shape, rows, cols); err != nil {
		return nil, err
	}
	wantElems, ok := checked.MulInt(rows, cols)
	if !ok {
		return nil, fmt.Errorf("%s element count overflow", name)
	}
	wantBytes, ok := checked.MulInt(wantElems, 2)
	if !ok {
		return nil, fmt.Errorf("%s byte count overflow", name)
	}
	if len(raw) != wantBytes {
		return nil, fmt.Errorf("%s raw len=%d want %d", name, len(raw), wantBytes)
	}
	return raw, nil
}

func rejectFrozenGPUTensorIfPresent(source weights.Source, name string) error {
	if _, _, _, err := source.GetRaw(name); err == nil {
		return fmt.Errorf("unsupported tensor %s: attention bias is not supported", name)
	} else if !strings.HasSuffix(err.Error(), "not found") && !strings.HasSuffix(err.Error(), "not in weight map") {
		return fmt.Errorf("inspect unsupported tensor %s: %w", name, err)
	}
	return nil
}

func frozenGPUExpectVector(name string, shape []int, n int) error {
	if len(shape) != 1 || shape[0] != n {
		return fmt.Errorf("%s shape=%v want [%d]", name, shape, n)
	}
	return nil
}

func frozenGPUExpectShape(name string, shape []int, dims ...int) error {
	if len(shape) != len(dims) {
		return fmt.Errorf("%s shape=%v want %v", name, shape, dims)
	}
	for i := range dims {
		if shape[i] != dims[i] {
			return fmt.Errorf("%s shape=%v want %v", name, shape, dims)
		}
	}
	return nil
}

func estimateFrozenGPUBytes(cfg LlamaConfig, host *frozenGPUHostModel, maxTokens int) (frozenGPUBytePlan, error) {
	var plan frozenGPUBytePlan
	var err error
	plan.matrix = host.matrixBytes
	plan.norm = host.normBytes
	kvDim, ok := checked.MulInt(cfg.NumKVHeads, cfg.HeadDim)
	if !ok {
		return plan, fmt.Errorf("KV dimension overflow")
	}
	_ = kvDim // K/V live in the shared prompt scratch, not persistent decode caches.
	ropeFreqs := frozenQwenRoPE(maxTokens, cfg.HeadDim, cfg.RopeTheta)
	plan.rope, err = frozenGPUVectorBytes(len(ropeFreqs))
	if err != nil {
		return plan, err
	}
	qDim, ok := checked.MulInt(cfg.NumHeads, cfg.HeadDim)
	if !ok {
		return plan, fmt.Errorf("Q dimension overflow")
	}
	workElems := 0
	for _, n := range []int{cfg.HiddenSize, cfg.HiddenSize, cfg.HiddenSize, qDim, qDim, kvDim, kvDim, kvDim, qDim, cfg.HiddenSize, cfg.Intermediate, cfg.Intermediate, cfg.HiddenSize} {
		workElems, ok = checked.AddInt(workElems, n)
		if !ok {
			return plan, fmt.Errorf("work buffer element overflow")
		}
	}
	workElems, ok = checked.MulInt(workElems, maxTokens)
	if !ok {
		return plan, fmt.Errorf("batch scratch overflow")
	}
	workBytes, err := frozenGPUVectorBytes(workElems)
	if err != nil {
		return plan, err
	}
	scratchBytes, err := frozenGPUVectorBytes(host.maxMatrixElem)
	if err != nil {
		return plan, err
	}
	plan.scratch, err = frozenGPUAddBytes(workBytes, scratchBytes)
	if err != nil {
		return plan, err
	}
	plan.total, err = frozenGPUAddBytes(plan.matrix, plan.norm)
	if err != nil {
		return plan, err
	}
	plan.total, err = frozenGPUAddBytes(plan.total, plan.kv)
	if err != nil {
		return plan, err
	}
	plan.total, err = frozenGPUAddBytes(plan.total, plan.rope)
	if err != nil {
		return plan, err
	}
	plan.total, err = frozenGPUAddBytes(plan.total, plan.scratch)
	if err != nil {
		return plan, err
	}
	return plan, nil
}

func checkFrozenGPUBudget(totalBytes, freeBytes uint64, options FrozenGPUOptions) error {
	if options.BudgetBytes > 0 && totalBytes > options.BudgetBytes {
		return fmt.Errorf("frozen GPU encoder needs %d bytes, budget is %d", totalBytes, options.BudgetBytes)
	}
	if freeBytes == 0 {
		return fmt.Errorf("GPU free-memory measurement unavailable")
	}
	if freeBytes > 0 {
		if options.ReserveBytes >= freeBytes {
			return fmt.Errorf("GPU reserve %d leaves no free memory from %d", options.ReserveBytes, freeBytes)
		}
		if totalBytes > freeBytes-options.ReserveBytes {
			return fmt.Errorf("frozen GPU encoder needs %d bytes, only %d bytes available after reserve %d", totalBytes, freeBytes-options.ReserveBytes, options.ReserveBytes)
		}
	}
	return nil
}

func (e *FrozenGPUEncoder) allocateFrozenGPU(host *frozenGPUHostModel, plan frozenGPUBytePlan) error {
	h := e.cfg.HiddenSize
	qDim, _ := checked.MulInt(e.cfg.NumHeads, e.cfg.HeadDim)
	kvDim, _ := checked.MulInt(e.cfg.NumKVHeads, e.cfg.HeadDim)
	var err error
	if e.finalNorm, err = uploadFrozenGPUVector(host.finalNorm); err != nil {
		return fmt.Errorf("upload model.norm.weight: %w", err)
	}
	ropeFreqs := frozenQwenRoPE(e.maxTokens, e.cfg.HeadDim, e.cfg.RopeTheta)
	if e.ropeTable, err = newFrozenGPUDevBufUpload(ropeFreqs); err != nil {
		return fmt.Errorf("upload rope table: %w", err)
	}
	if e.weightScratch, err = nvidia.Malloc(host.maxMatrixElem); err != nil {
		return fmt.Errorf("alloc weight scratch: %w", err)
	}
	for _, alloc := range []struct {
		dst  **nvidia.DevBuf
		n    int
		name string
	}{
		{&e.hidden, h, "hidden"},
		{&e.residual, h, "residual"},
		{&e.normed, h, "normed"},
		{&e.q, qDim, "q"},
		{&e.qNormed, qDim, "q_normed"},
		{&e.k, kvDim, "k"},
		{&e.kNormed, kvDim, "k_normed"},
		{&e.v, kvDim, "v"},
		{&e.attnOut, qDim, "attn_out"},
		{&e.oOut, h, "o_out"},
		{&e.gate, e.cfg.Intermediate, "gate"},
		{&e.up, e.cfg.Intermediate, "up"},
		{&e.down, h, "down"},
	} {
		buf, err := nvidia.NewDevBufGPU(alloc.n * e.maxTokens)
		if err != nil {
			return fmt.Errorf("alloc %s: %w", alloc.name, err)
		}
		*alloc.dst = buf
	}
	e.layers = make([]frozenGPULayer, len(host.layers))
	for i := range host.layers {
		hl := host.layers[i]
		gl := &e.layers[i]
		gl.headDim = hl.headDim
		gl.qDim = hl.qDim
		gl.kvDim = hl.kvDim
		if gl.inputNorm, err = uploadFrozenGPUVector(hl.inputNorm); err != nil {
			return fmt.Errorf("upload layer %d input norm: %w", i, err)
		}
		if gl.postNorm, err = uploadFrozenGPUVector(hl.postNorm); err != nil {
			return fmt.Errorf("upload layer %d post norm: %w", i, err)
		}
		if gl.qNorm, err = uploadFrozenGPUVector(hl.qNorm); err != nil {
			return fmt.Errorf("upload layer %d q norm: %w", i, err)
		}
		if gl.kNorm, err = uploadFrozenGPUVector(hl.kNorm); err != nil {
			return fmt.Errorf("upload layer %d k norm: %w", i, err)
		}
		for _, up := range []struct {
			raw  []byte
			dst  **nvidia.Buffer
			name string
		}{
			{hl.qProj, &gl.qProj, "q_proj"},
			{hl.kProj, &gl.kProj, "k_proj"},
			{hl.vProj, &gl.vProj, "v_proj"},
			{hl.oProj, &gl.oProj, "o_proj"},
			{hl.gateProj, &gl.gateProj, "gate_proj"},
			{hl.upProj, &gl.upProj, "up_proj"},
			{hl.downProj, &gl.downProj, "down_proj"},
		} {
			buf, err := uploadFrozenGPURawBF16(up.raw)
			if err != nil {
				return fmt.Errorf("upload layer %d %s: %w", i, up.name, err)
			}
			*up.dst = buf
		}

	}
	e.embedRaw = host.embedRaw
	e.embedRow = make([]float32, h)
	e.stats.AllocatedBytes = plan.total
	return nil
}

func uploadFrozenGPUVector(data []float32) (*nvidia.Buffer, error) {
	buf, err := nvidia.Malloc(len(data))
	if err != nil {
		return nil, err
	}
	if err := buf.Upload(data); err != nil {
		buf.Free()
		return nil, err
	}
	return buf, nil
}

func uploadFrozenGPURawBF16(raw []byte) (*nvidia.Buffer, error) {
	buf, err := nvidia.Malloc(frozenGPUFloat32SlotsForBytes(len(raw)))
	if err != nil {
		return nil, err
	}
	if err := buf.UploadBytes(raw); err != nil {
		buf.Free()
		return nil, err
	}
	return buf, nil
}

func newFrozenGPUDevBufUpload(data []float32) (*nvidia.DevBuf, error) {
	buf, err := nvidia.NewDevBufGPU(len(data))
	if err != nil {
		return nil, err
	}
	if len(data) > 0 {
		if err := buf.GPUBuffer().Upload(data); err != nil {
			buf.Free()
			return nil, err
		}
	}
	return buf, nil
}

func (e *FrozenGPUEncoder) uploadEmbedding(tokenID, pos int) error {
	h := e.cfg.HiddenSize
	rowElems, ok := checked.MulInt(tokenID, h)
	if !ok {
		return fmt.Errorf("embedding offset overflow")
	}
	rowBytes, ok := checked.MulInt(rowElems, 2)
	if !ok {
		return fmt.Errorf("embedding byte offset overflow")
	}
	rowLen, ok := checked.MulInt(h, 2)
	if !ok {
		return fmt.Errorf("embedding row byte length overflow")
	}
	end, ok := checked.AddInt(rowBytes, rowLen)
	if !ok || end > len(e.embedRaw) {
		return fmt.Errorf("embedding row out of bounds")
	}
	raw := e.embedRaw[rowBytes:end]
	for i := 0; i < h; i++ {
		bits := uint32(raw[i*2]) | uint32(raw[i*2+1])<<8
		e.embedRow[i] = math.Float32frombits(bits << 16)
	}
	return e.hidden.Slice(pos*h, h).GPUBuffer().Upload(e.embedRow)
}

func (e *FrozenGPUEncoder) normRows(out, input *nvidia.DevBuf, weight *nvidia.Buffer, rows, width int) error {
	for row := 0; row < rows; row++ {
		if err := nvidia.F32RMSNormBuffer(out.Slice(row*width, width).GPUBuffer(), input.Slice(row*width, width).GPUBuffer(), weight, width, float32(e.cfg.RMSNormEps)); err != nil {
			return err
		}
	}
	return nil
}

func (e *FrozenGPUEncoder) forwardLayer(layerIdx, batch int) error {
	l := &e.layers[layerIdx]
	h, q, k, inter := e.cfg.HiddenSize, l.qDim, l.kvDim, e.cfg.Intermediate
	if err := frozenGPUCopy(e.residual, e.hidden, batch*h); err != nil {
		return err
	}
	if err := e.normRows(e.normed, e.hidden, l.inputNorm, batch, h); err != nil {
		return err
	}
	for _, p := range []struct {
		out    *nvidia.DevBuf
		weight *nvidia.Buffer
		rows   int
	}{{e.q, l.qProj, q}, {e.k, l.kProj, k}, {e.v, l.vProj, k}} {
		if err := e.projectBF16(p.out, e.normed, p.weight, batch, p.rows, h); err != nil {
			return err
		}
	}
	if err := e.normRows(e.qNormed, e.q, l.qNorm, batch*e.cfg.NumHeads, l.headDim); err != nil {
		return err
	}
	if err := e.normRows(e.kNormed, e.k, l.kNorm, batch*e.cfg.NumKVHeads, l.headDim); err != nil {
		return err
	}
	for pos := 0; pos < batch; pos++ {
		if !nvidia.DevRoPE(e.qNormed.Slice(pos*q, q), e.ropeTable, pos, e.cfg.NumHeads, l.headDim) || !nvidia.DevRoPE(e.kNormed.Slice(pos*k, k), e.ropeTable, pos, e.cfg.NumKVHeads, l.headDim) {
			return fmt.Errorf("GPU RoPE rejected")
		}
		if !nvidia.DevAttentionOK(e.attnOut.Slice(pos*q, q), e.qNormed.Slice(pos*q, q), e.kNormed.Slice(0, (pos+1)*k), e.v.Slice(0, (pos+1)*k), pos+1, e.cfg.NumHeads, e.cfg.NumKVHeads, l.headDim, attentionScale(e.cfg, l.headDim)) {
			return fmt.Errorf("GPU attention rejected")
		}
	}
	if err := e.projectBF16(e.oOut, e.attnOut, l.oProj, batch, h, q); err != nil {
		return err
	}
	out := e.hidden.Slice(0, batch*h)
	nvidia.DevAdd(out, e.residual.Slice(0, batch*h), e.oOut.Slice(0, batch*h))
	if !out.OnGPU() {
		return fmt.Errorf("GPU residual unavailable")
	}
	if err := frozenGPUCopy(e.residual, e.hidden, batch*h); err != nil {
		return err
	}
	if err := e.normRows(e.normed, e.hidden, l.postNorm, batch, h); err != nil {
		return err
	}
	if err := e.projectBF16(e.gate, e.normed, l.gateProj, batch, inter, h); err != nil {
		return err
	}
	if err := e.projectBF16(e.up, e.normed, l.upProj, batch, inter, h); err != nil {
		return err
	}
	gate := e.gate.Slice(0, batch*inter)
	nvidia.DevSiLUMul(gate, gate, e.up.Slice(0, batch*inter))
	if !gate.OnGPU() {
		return fmt.Errorf("GPU SiLU unavailable")
	}
	if err := e.projectBF16(e.down, e.gate, l.downProj, batch, h, inter); err != nil {
		return err
	}
	out = e.hidden.Slice(0, batch*h)
	nvidia.DevAdd(out, e.residual.Slice(0, batch*h), e.down.Slice(0, batch*h))
	if !out.OnGPU() {
		return fmt.Errorf("GPU residual unavailable")
	}
	return nil
}

func (e *FrozenGPUEncoder) projectBF16(out, input *nvidia.DevBuf, weight *nvidia.Buffer, batch, rows, cols int) error {
	if out == nil || input == nil || weight == nil || !out.OnGPU() || !input.OnGPU() {
		return fmt.Errorf("projection buffers not resident")
	}
	if err := nvidia.WidenBF16Transpose(e.weightScratch, weight, rows, cols); err != nil {
		return err
	}
	return nvidia.SgemmCompensated(batch, rows, cols, 1, input.GPUBuffer(), e.weightScratch, out.GPUBuffer())
}

func frozenGPUCopy(dst, src *nvidia.DevBuf, n int) error {
	if dst == nil || src == nil || !dst.OnGPU() || !src.OnGPU() {
		return fmt.Errorf("copy buffers not resident on GPU")
	}
	bytes, err := frozenGPUVectorBytesInt(n)
	if err != nil {
		return err
	}
	return nvidia.CopyDtoD(dst.GPUBuffer().Ptr, src.GPUBuffer().Ptr, uint64(bytes))
}

func frozenGPUFloat32SlotsForBytes(n int) int {
	if n <= 0 {
		return 0
	}
	if n%4 == 0 {
		return n / 4
	}
	return n/4 + 1
}

func frozenGPUPaddedRawBytes(raw []byte) (uint64, error) {
	return frozenGPUVectorBytes(frozenGPUFloat32SlotsForBytes(len(raw)))
}

func frozenGPUVectorBytes(n int) (uint64, error) {
	if n < 0 {
		return 0, fmt.Errorf("negative element count %d", n)
	}
	if n > int(^uint(0)>>1)/4 {
		return 0, fmt.Errorf("byte size overflow for %d float32 values", n)
	}
	return uint64(n * 4), nil
}

func frozenGPUVectorBytesInt(n int) (int, error) {
	if n < 0 || n > int(^uint(0)>>1)/4 {
		return 0, fmt.Errorf("byte size overflow for %d float32 values", n)
	}
	return n * 4, nil
}

func frozenGPUAddBytes(a, b uint64) (uint64, error) {
	if ^uint64(0)-a < b {
		return 0, fmt.Errorf("byte count overflow")
	}
	return a + b, nil
}

func frozenGPUMulBytes(a, b uint64) (uint64, error) {
	if a == 0 || b == 0 {
		return 0, nil
	}
	if a > ^uint64(0)/b {
		return 0, fmt.Errorf("byte count overflow")
	}
	return a * b, nil
}
