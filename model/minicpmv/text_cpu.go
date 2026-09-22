package minicpmv

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"strings"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/internal/checked"
	"github.com/rcarmo/go-pherence/loader/config"
	"github.com/rcarmo/go-pherence/loader/weights"
)

// Float32TensorSource is the owned F32 tensor surface used by the text CPU
// loader. safetensors.File and safetensors.ShardedFile return owned decoded
// values and satisfy this contract.
type Float32TensorSource interface {
	GetFloat32(name string) ([]float32, []int, error)
}

type textCPUConfig struct {
	family, modelPrefix, headName             string
	hidden, layers, heads, kvHeads, headDim   int
	intermediate, vocab, maxPosition          int
	eps, theta, embeddingScale, residualScale float32
	logitScale                                float32
	qkvBias, tied                             bool
}

type textLinear struct {
	weight        []float32
	bias          []float32
	inDim, outDim int
}

type textCPULayer struct {
	inputNorm, postNorm []float32
	q, k, v, o          textLinear
	gate, up, down      textLinear
}

// TextCPUState owns all mutable KV state for one request. A state may be cloned
// and resumed independently; TextCPU itself contains immutable owned weights.
type TextCPUState struct {
	K   [][]float32
	V   [][]float32
	Pos int
}

func (s TextCPUState) Clone() TextCPUState {
	out := TextCPUState{K: make([][]float32, len(s.K)), V: make([][]float32, len(s.V)), Pos: s.Pos}
	for i := range s.K {
		out.K[i] = append([]float32(nil), s.K[i]...)
	}
	for i := range s.V {
		out.V[i] = append([]float32(nil), s.V[i]...)
	}
	return out
}

// TextTokenResult is the synthetic/released-fixture comparison boundary for
// one token. Hidden is post-final-norm; Logits are unmodified F32 LM-head
// output. Both slices are newly owned by the caller.
type TextTokenResult struct {
	Hidden []float32
	Logits []float32
}

// TextCPU binds the dense MiniCPM, Qwen2, or Mistral decoder tensor layout.
// It is safe for concurrent use with separate TextCPUState values.
type TextCPU struct {
	cfg       textCPUConfig
	embedding []float32
	layers    []textCPULayer
	norm      []float32
	lmHead    []float32
}

// LoadTextCPUFromDir loads an unsharded or indexed safetensors checkpoint.
func LoadTextCPUFromDir(dir string, cfg config.MiniCPMVConfig) (*TextCPU, error) {
	src, err := weights.OpenSafetensors(dir)
	if err != nil {
		return nil, err
	}
	defer src.Close()
	return LoadTextCPU(src, cfg)
}

// LoadTextCPU transactionally binds an exact dense text backbone. It rejects
// unsupported attention/activation/RoPE policies instead of approximating
// them, and requires PyTorch [out,in] matrix orientation.
func LoadTextCPU(src Float32TensorSource, cfg config.MiniCPMVConfig) (*TextCPU, error) {
	if src == nil {
		return nil, fmt.Errorf("nil MiniCPM-V/O text tensor source")
	}
	c, err := normalizeTextCPUConfig(cfg)
	if err != nil {
		return nil, err
	}
	m := &TextCPU{cfg: c, layers: make([]textCPULayer, c.layers)}
	if m.embedding, err = loadTextTensor(src, c.modelPrefix+".embed_tokens.weight", []int{c.vocab, c.hidden}); err != nil {
		return nil, err
	}
	if m.norm, err = loadTextTensor(src, c.modelPrefix+".norm.weight", []int{c.hidden}); err != nil {
		return nil, err
	}
	if c.tied {
		m.lmHead = m.embedding
	} else if m.lmHead, err = loadTextTensor(src, c.headName, []int{c.vocab, c.hidden}); err != nil {
		return nil, err
	}
	queryWidth, okQ := checked.MulInt(c.heads, c.headDim)
	kvWidth, okKV := checked.MulInt(c.kvHeads, c.headDim)
	if !okQ || !okKV {
		return nil, fmt.Errorf("MiniCPM-V/O text projection dimensions overflow")
	}
	for i := range m.layers {
		prefix := fmt.Sprintf("%s.layers.%d", c.modelPrefix, i)
		l := &m.layers[i]
		if l.inputNorm, err = loadTextTensor(src, prefix+".input_layernorm.weight", []int{c.hidden}); err != nil {
			return nil, err
		}
		if l.postNorm, err = loadTextTensor(src, prefix+".post_attention_layernorm.weight", []int{c.hidden}); err != nil {
			return nil, err
		}
		if l.q, err = loadTextLinear(src, prefix+".self_attn.q_proj", c.hidden, queryWidth, c.qkvBias); err != nil {
			return nil, err
		}
		if l.k, err = loadTextLinear(src, prefix+".self_attn.k_proj", c.hidden, kvWidth, c.qkvBias); err != nil {
			return nil, err
		}
		if l.v, err = loadTextLinear(src, prefix+".self_attn.v_proj", c.hidden, kvWidth, c.qkvBias); err != nil {
			return nil, err
		}
		if l.o, err = loadTextLinear(src, prefix+".self_attn.o_proj", queryWidth, c.hidden, false); err != nil {
			return nil, err
		}
		if l.gate, err = loadTextLinear(src, prefix+".mlp.gate_proj", c.hidden, c.intermediate, false); err != nil {
			return nil, err
		}
		if l.up, err = loadTextLinear(src, prefix+".mlp.up_proj", c.hidden, c.intermediate, false); err != nil {
			return nil, err
		}
		if l.down, err = loadTextLinear(src, prefix+".mlp.down_proj", c.intermediate, c.hidden, false); err != nil {
			return nil, err
		}
	}
	return m, nil
}

func normalizeTextCPUConfig(cfg config.MiniCPMVConfig) (textCPUConfig, error) {
	s := cfg.MiniCPMVSummary()
	c := textCPUConfig{
		family: strings.ToLower(strings.TrimSpace(s.TextModelType)), hidden: s.HiddenSize,
		layers: s.Layers, heads: s.Heads, kvHeads: s.KVHeads, headDim: s.HeadDim,
		intermediate: s.IntermediateSize, vocab: s.VocabSize,
	}
	text := cfg.TextConfig
	if c.family == "" {
		switch cfg.ModelType {
		case "minicpmo", "minicpm_o", "minicpm-o":
			c.family = "qwen2"
		case "minicpmv", "minicpm_v", "minicpm-v":
			if cfg.ScaleEmb != 0 || cfg.ScaleDepth != 0 || cfg.DimModelBase != 0 {
				c.family = "minicpm"
			} else {
				c.family = "qwen2"
			}
		case "omnilmm":
			c.family = "mistral"
		}
	}
	switch c.family {
	case "minicpm", "qwen2":
		c.modelPrefix, c.headName = "llm.model", "llm.lm_head.weight"
	case "mistral":
		c.modelPrefix, c.headName = "model", "lm_head.weight"
	default:
		return c, fmt.Errorf("unsupported MiniCPM-V/O text model type %q", c.family)
	}
	if c.hidden <= 0 || c.layers <= 0 || c.heads <= 0 || c.kvHeads <= 0 || c.headDim <= 0 || c.intermediate <= 0 || c.vocab <= 0 || c.heads%c.kvHeads != 0 {
		return c, fmt.Errorf("invalid MiniCPM-V/O text dimensions hidden=%d layers=%d heads=%d kv_heads=%d head_dim=%d intermediate=%d vocab=%d", c.hidden, c.layers, c.heads, c.kvHeads, c.headDim, c.intermediate, c.vocab)
	}
	queryWidth, ok := checked.MulInt(c.heads, c.headDim)
	if !ok || queryWidth != c.hidden || c.headDim%2 != 0 {
		return c, fmt.Errorf("unsupported MiniCPM-V/O text attention dimensions hidden=%d query_width=%d head_dim=%d", c.hidden, queryWidth, c.headDim)
	}
	c.maxPosition = cfg.MaxPositionEmbeds
	c.eps = float32(cfg.RMSNormEps)
	c.theta = float32(cfg.RopeTheta)
	hiddenAct := cfg.HiddenAct
	attentionBias := cfg.AttentionBias
	ropeScaling := cfg.RopeScaling
	c.tied = cfg.TieWordEmbeddings
	c.embeddingScale = 1
	c.residualScale = 1
	c.logitScale = 1
	if text != nil {
		if text.MaxPositionEmbeds > 0 {
			c.maxPosition = text.MaxPositionEmbeds
		}
		if text.RMSNormEps > 0 {
			c.eps = float32(text.RMSNormEps)
		}
		if text.RopeTheta > 0 {
			c.theta = float32(text.RopeTheta)
		}
		if text.HiddenAct != "" {
			hiddenAct = text.HiddenAct
		}
		if text.AttentionBias != nil {
			attentionBias = text.AttentionBias
		}
		if len(text.RopeScaling) != 0 {
			ropeScaling = text.RopeScaling
		}
		if text.TieWordEmbeddings != nil {
			c.tied = *text.TieWordEmbeddings
		}
	}
	if c.maxPosition <= 0 {
		return c, fmt.Errorf("invalid MiniCPM-V/O max_position_embeddings=%d", c.maxPosition)
	}
	if c.eps <= 0 || !finite(float64(c.eps)) {
		return c, fmt.Errorf("invalid MiniCPM-V/O rms_norm_eps=%g", c.eps)
	}
	if c.theta == 0 {
		c.theta = 10000
	}
	if c.theta <= 0 || !finite(float64(c.theta)) {
		return c, fmt.Errorf("invalid MiniCPM-V/O rope_theta=%g", c.theta)
	}
	if hiddenAct != "" && strings.ToLower(hiddenAct) != "silu" {
		return c, fmt.Errorf("unsupported MiniCPM-V/O hidden_act=%q", hiddenAct)
	}
	if !nullJSON(ropeScaling) {
		return c, fmt.Errorf("unsupported MiniCPM-V/O rope_scaling=%s", strings.TrimSpace(string(ropeScaling)))
	}
	// Qwen2 publishes biases for Q/K/V but not O. MiniCPM and Mistral are
	// bias-free unless their config explicitly opts in.
	c.qkvBias = c.family == "qwen2"
	if attentionBias != nil {
		c.qkvBias = *attentionBias
	}
	if cfg.UseSlidingWindow || (text != nil && text.UseSlidingWindow) {
		return c, fmt.Errorf("unsupported MiniCPM-V/O sliding-window text attention")
	}
	if c.family == "minicpm" {
		scaleEmb := cfg.ScaleEmb
		scaleDepth := cfg.ScaleDepth
		dimModelBase := cfg.DimModelBase
		if text != nil {
			if text.ScaleEmb != 0 {
				scaleEmb = text.ScaleEmb
			}
			if text.ScaleDepth != 0 {
				scaleDepth = text.ScaleDepth
			}
			if text.DimModelBase != 0 {
				dimModelBase = text.DimModelBase
			}
		}
		if scaleEmb == 0 {
			scaleEmb = 1
		}
		if scaleDepth == 0 {
			scaleDepth = 1
		}
		if dimModelBase == 0 {
			dimModelBase = c.hidden
		}
		c.embeddingScale = float32(scaleEmb)
		c.residualScale = float32(scaleDepth / math.Sqrt(float64(c.layers)))
		c.logitScale = float32(float64(dimModelBase) / float64(c.hidden))
		if !finite(float64(c.embeddingScale)) || !finite(float64(c.residualScale)) || !finite(float64(c.logitScale)) {
			return c, fmt.Errorf("invalid MiniCPM text scaling")
		}
	}
	return c, nil
}

func nullJSON(raw json.RawMessage) bool {
	trimmed := bytes.TrimSpace(raw)
	return len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null"))
}

func finite(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }

func loadTextLinear(src Float32TensorSource, prefix string, inDim, outDim int, bias bool) (textLinear, error) {
	weight, err := loadTextTensor(src, prefix+".weight", []int{outDim, inDim})
	if err != nil {
		return textLinear{}, err
	}
	l := textLinear{weight: weight, inDim: inDim, outDim: outDim}
	if bias {
		l.bias, err = loadTextTensor(src, prefix+".bias", []int{outDim})
		if err != nil {
			return textLinear{}, err
		}
	}
	return l, nil
}

func loadTextTensor(src Float32TensorSource, name string, want []int) ([]float32, error) {
	data, shape, err := src.GetFloat32(name)
	if err != nil {
		return nil, fmt.Errorf("load %s: %w", name, err)
	}
	if !equalTextShape(shape, want) {
		return nil, fmt.Errorf("load %s: shape=%v want %v", name, shape, want)
	}
	wantLen := 1
	for _, dim := range want {
		var ok bool
		wantLen, ok = checked.MulInt(wantLen, dim)
		if !ok {
			return nil, fmt.Errorf("load %s: shape product overflows", name)
		}
	}
	if len(data) != wantLen {
		return nil, fmt.Errorf("load %s: values=%d want %d", name, len(data), wantLen)
	}
	for i, value := range data {
		if !finite(float64(value)) {
			return nil, fmt.Errorf("load %s: non-finite value at %d", name, i)
		}
	}
	return data, nil
}

func equalTextShape(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func (l textLinear) forward(dst, input []float32) error {
	if len(dst) != l.outDim || len(input) != l.inDim || !simd.GemvRows(dst, input, l.weight, l.outDim, l.inDim) {
		return fmt.Errorf("invalid MiniCPM-V/O linear buffers out/in=%d/%d want %d/%d", len(dst), len(input), l.outDim, l.inDim)
	}
	if len(l.bias) != 0 {
		if len(l.bias) != len(dst) {
			return fmt.Errorf("invalid MiniCPM-V/O linear bias=%d want %d", len(l.bias), len(dst))
		}
		for i := range dst {
			dst[i] += l.bias[i]
		}
	}
	return nil
}

func (m *TextCPU) NewState() TextCPUState {
	if m == nil {
		return TextCPUState{}
	}
	return TextCPUState{K: make([][]float32, len(m.layers)), V: make([][]float32, len(m.layers))}
}

// GenerateFromEmbeddings implements TextBackbone for an already assembled
// text/image/audio embedding prefix. It performs deterministic greedy decode;
// returned token IDs and all mutable state are request-local.
func (m *TextCPU) GenerateFromEmbeddings(embeddings []float32, seqLen, hidden, maxNewTokens int) ([]int, error) {
	if m == nil {
		return nil, fmt.Errorf("nil MiniCPM-V/O CPU text backbone")
	}
	values, ok := checked.MulInt(seqLen, hidden)
	maxSequence, okSequence := checked.AddInt(seqLen, maxNewTokens)
	if seqLen <= 0 || hidden != m.cfg.hidden || maxNewTokens <= 0 || !ok || len(embeddings) != values || !okSequence || maxSequence > m.cfg.maxPosition {
		return nil, fmt.Errorf("invalid MiniCPM-V/O generation inputs values=%d sequence=%d hidden=%d max_new_tokens=%d", len(embeddings), seqLen, hidden, maxNewTokens)
	}
	state := m.NewState()
	var result TextTokenResult
	var err error
	for pos := 0; pos < seqLen; pos++ {
		result, state, err = m.ForwardEmbedding(embeddings[pos*hidden:(pos+1)*hidden], state)
		if err != nil {
			return nil, err
		}
	}
	generated := make([]int, 0, maxNewTokens)
	for len(generated) < maxNewTokens {
		token, err := greedyTextToken(result.Logits)
		if err != nil {
			return nil, err
		}
		generated = append(generated, token)
		if len(generated) == maxNewTokens {
			break
		}
		result, state, err = m.ForwardToken(uint32(token), state)
		if err != nil {
			return nil, err
		}
	}
	return generated, nil
}

func greedyTextToken(logits []float32) (int, error) {
	if len(logits) == 0 {
		return 0, fmt.Errorf("empty MiniCPM-V/O logits")
	}
	best := -1
	bestValue := float32(math.Inf(-1))
	for i, value := range logits {
		if !finite(float64(value)) {
			return 0, fmt.Errorf("non-finite MiniCPM-V/O logit at %d", i)
		}
		if best < 0 || value > bestValue {
			best, bestValue = i, value
		}
	}
	return best, nil
}

// ForwardToken embeds one token and executes one causal decoder step.
func (m *TextCPU) ForwardToken(token uint32, state TextCPUState) (TextTokenResult, TextCPUState, error) {
	if m == nil {
		return TextTokenResult{}, state, fmt.Errorf("nil MiniCPM-V/O CPU text backbone")
	}
	if int(token) >= m.cfg.vocab {
		return TextTokenResult{}, state, fmt.Errorf("invalid MiniCPM-V/O text token=%d vocab=%d", token, m.cfg.vocab)
	}
	start := int(token) * m.cfg.hidden
	input := append([]float32(nil), m.embedding[start:start+m.cfg.hidden]...)
	if m.cfg.embeddingScale != 1 {
		for i := range input {
			input[i] *= m.cfg.embeddingScale
		}
	}
	return m.ForwardEmbedding(input, state)
}

// ForwardEmbedding executes one caller-provided language-sized embedding. It
// copies the input and commits KV changes only in the returned state.
func (m *TextCPU) ForwardEmbedding(embedding []float32, state TextCPUState) (TextTokenResult, TextCPUState, error) {
	if m == nil {
		return TextTokenResult{}, state, fmt.Errorf("nil MiniCPM-V/O CPU text backbone")
	}
	if len(embedding) != m.cfg.hidden {
		return TextTokenResult{}, state, fmt.Errorf("invalid MiniCPM-V/O embedding=%d want %d", len(embedding), m.cfg.hidden)
	}
	for i, value := range embedding {
		if !finite(float64(value)) {
			return TextTokenResult{}, state, fmt.Errorf("non-finite MiniCPM-V/O embedding at %d", i)
		}
	}
	if state.Pos < 0 || state.Pos >= m.cfg.maxPosition || len(state.K) != len(m.layers) || len(state.V) != len(m.layers) {
		return TextTokenResult{}, state, fmt.Errorf("invalid MiniCPM-V/O text state position/layers=%d/%d/%d", state.Pos, len(state.K), len(state.V))
	}
	kvWidth, ok := checked.MulInt(m.cfg.kvHeads, m.cfg.headDim)
	if !ok {
		return TextTokenResult{}, state, fmt.Errorf("MiniCPM-V/O KV width overflows")
	}
	wantPast, ok := checked.MulInt(state.Pos, kvWidth)
	if !ok {
		return TextTokenResult{}, state, fmt.Errorf("MiniCPM-V/O KV state length overflows")
	}
	for i := range m.layers {
		if len(state.K[i]) != wantPast || len(state.V[i]) != wantPast {
			return TextTokenResult{}, state, fmt.Errorf("invalid MiniCPM-V/O layer %d KV lengths=%d/%d want %d", i, len(state.K[i]), len(state.V[i]), wantPast)
		}
	}
	next := state.Clone()
	hidden := append([]float32(nil), embedding...)
	rope := simd.BuildRoPEFreqs(state.Pos+1, m.cfg.headDim/2, m.cfg.headDim, float64(m.cfg.theta))
	if len(rope) == 0 {
		return TextTokenResult{}, state, fmt.Errorf("MiniCPM-V/O text RoPE table is empty")
	}
	for i := range m.layers {
		var err error
		hidden, next.K[i], next.V[i], err = m.layers[i].forward(hidden, next.K[i], next.V[i], state.Pos, rope, m.cfg)
		if err != nil {
			return TextTokenResult{}, state, fmt.Errorf("MiniCPM-V/O text layer %d position %d: %w", i, state.Pos, err)
		}
	}
	if !simd.RMSNormTo(hidden, m.norm, m.cfg.eps) {
		return TextTokenResult{}, state, fmt.Errorf("MiniCPM-V/O final RMSNorm failed")
	}
	logits := make([]float32, m.cfg.vocab)
	if !simd.GemvRows(logits, hidden, m.lmHead, m.cfg.vocab, m.cfg.hidden) {
		return TextTokenResult{}, state, fmt.Errorf("MiniCPM-V/O LM head failed")
	}
	if m.cfg.logitScale != 1 {
		for i := range logits {
			logits[i] *= m.cfg.logitScale
		}
	}
	if err := validateFiniteTextOutput("hidden", hidden); err != nil {
		return TextTokenResult{}, state, err
	}
	if err := validateFiniteTextOutput("logit", logits); err != nil {
		return TextTokenResult{}, state, err
	}
	next.Pos++
	return TextTokenResult{Hidden: hidden, Logits: logits}, next, nil
}

func (l *textCPULayer) forward(input, pastK, pastV []float32, pos int, rope []float32, cfg textCPUConfig) ([]float32, []float32, []float32, error) {
	normed := append([]float32(nil), input...)
	if !simd.RMSNormTo(normed, l.inputNorm, cfg.eps) {
		return nil, nil, nil, fmt.Errorf("input RMSNorm failed")
	}
	queryWidth := cfg.heads * cfg.headDim
	kvWidth := cfg.kvHeads * cfg.headDim
	q := make([]float32, queryWidth)
	k := make([]float32, kvWidth)
	v := make([]float32, kvWidth)
	if err := l.q.forward(q, normed); err != nil {
		return nil, nil, nil, err
	}
	if err := l.k.forward(k, normed); err != nil {
		return nil, nil, nil, err
	}
	if err := l.v.forward(v, normed); err != nil {
		return nil, nil, nil, err
	}
	if !simd.ApplyRoPETo(q, rope, pos, cfg.heads, cfg.headDim) || !simd.ApplyRoPETo(k, rope, pos, cfg.kvHeads, cfg.headDim) {
		return nil, nil, nil, fmt.Errorf("RoPE failed")
	}
	kAll := append(pastK, k...)
	vAll := append(pastV, v...)
	attention := make([]float32, queryWidth)
	scores := make([]float32, pos+1)
	if !simd.GQAAttentionScaleTo(attention, scores, q, kAll, vAll, pos+1, cfg.heads, cfg.kvHeads, cfg.headDim, float32(1/math.Sqrt(float64(cfg.headDim)))) {
		return nil, nil, nil, fmt.Errorf("GQA attention failed")
	}
	projected := make([]float32, cfg.hidden)
	if err := l.o.forward(projected, attention); err != nil {
		return nil, nil, nil, err
	}
	residual := make([]float32, cfg.hidden)
	if !simd.VecScaleAddTo(residual, input, projected, cfg.residualScale) {
		return nil, nil, nil, fmt.Errorf("attention residual failed")
	}
	mlpInput := append([]float32(nil), residual...)
	if !simd.RMSNormTo(mlpInput, l.postNorm, cfg.eps) {
		return nil, nil, nil, fmt.Errorf("post-attention RMSNorm failed")
	}
	gate := make([]float32, cfg.intermediate)
	up := make([]float32, cfg.intermediate)
	if err := l.gate.forward(gate, mlpInput); err != nil {
		return nil, nil, nil, err
	}
	if err := l.up.forward(up, mlpInput); err != nil {
		return nil, nil, nil, err
	}
	if !simd.SiLUMulTo(gate, gate, up) {
		return nil, nil, nil, fmt.Errorf("MLP SwiGLU failed")
	}
	down := make([]float32, cfg.hidden)
	if err := l.down.forward(down, gate); err != nil {
		return nil, nil, nil, err
	}
	out := make([]float32, cfg.hidden)
	if !simd.VecScaleAddTo(out, residual, down, cfg.residualScale) {
		return nil, nil, nil, fmt.Errorf("MLP residual failed")
	}
	return out, kAll, vAll, nil
}

func validateFiniteTextOutput(kind string, values []float32) error {
	for i, value := range values {
		if !finite(float64(value)) {
			return fmt.Errorf("non-finite MiniCPM-V/O %s at %d", kind, i)
		}
	}
	return nil
}
