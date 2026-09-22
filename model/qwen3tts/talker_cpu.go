package qwen3tts

import (
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/loader/weights"
)

// Float32TensorSource is the owned F32 tensor surface used by the CPU Talker
// loader. Implementations return caller-owned values; safetensors.File and
// safetensors.ShardedFile satisfy that contract.
type Float32TensorSource interface {
	GetFloat32(name string) ([]float32, []int, error)
}

type talkerLinear struct {
	weight        []float32
	bias          []float32
	inDim, outDim int
}

type talkerCPULayer struct {
	inputNorm, postNorm []float32
	q, k, v, o          talkerLinear
	qNorm, kNorm        []float32
	gate, up, down      talkerLinear
}

// TalkerCPU owns the decoded F32 weights for the Qwen3-TTS Talker reference
// path. It is safe for concurrent calls because all mutable KV and scratch state
// is request-local.
type TalkerCPU struct {
	cfg                              ParsedConfig
	textEmbedding, codecEmbedding    []float32
	textProjection1, textProjection2 talkerLinear
	layers                           []talkerCPULayer
	norm                             []float32
	codecHead                        talkerLinear
}

// TalkerPrefillResult exposes the deterministic Talker prefill boundary used by
// independent fixture comparison. Logits are returned before token suppression.
type TalkerPrefillResult struct {
	Hidden        []float32
	Logits        []float32
	SemanticToken uint32
	PrefillTokens int
}

// LoadTalkerCPUFromDir loads an unsharded or indexed safetensors checkpoint.
// All required Talker tensors are copied before the source is closed.
func LoadTalkerCPUFromDir(dir string, cfg ParsedConfig) (*TalkerCPU, error) {
	src, err := weights.OpenSafetensors(dir)
	if err != nil {
		return nil, err
	}
	defer src.Close()
	return LoadTalkerCPU(src, cfg)
}

// LoadTalkerCPU binds the dense Talker tensors transactionally. It accepts the
// official Hugging Face tensor names and rejects missing or transposed shapes.
func LoadTalkerCPU(src Float32TensorSource, cfg ParsedConfig) (*TalkerCPU, error) {
	if src == nil {
		return nil, fmt.Errorf("nil Qwen3-TTS Talker tensor source")
	}
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if cfg.ModelType != CustomVoice {
		return nil, fmt.Errorf("Qwen3-TTS CPU Talker supports custom_voice, got %s", cfg.ModelType)
	}
	if cfg.HasMRoPESection {
		pairs := sizeSum(cfg.MRoPESection[0], cfg.MRoPESection[1], cfg.MRoPESection[2])
		if pairs != cfg.TalkerHeadDim/2 {
			return nil, fmt.Errorf("Qwen3-TTS Talker mrope sections sum=%d want head_dim/2=%d", pairs, cfg.TalkerHeadDim/2)
		}
	}
	m := &TalkerCPU{cfg: cfg, layers: make([]talkerCPULayer, cfg.TalkerNumHiddenLayers)}
	var err error
	if m.textEmbedding, err = loadTalkerTensor(src, "talker.model.text_embedding.weight", []int{cfg.TalkerTextVocabSize, cfg.TalkerTextHiddenSize}); err != nil {
		return nil, err
	}
	if m.codecEmbedding, err = loadTalkerTensor(src, "talker.model.codec_embedding.weight", []int{cfg.TalkerVocabSize, cfg.TalkerHiddenSize}); err != nil {
		return nil, err
	}
	if m.textProjection1, err = loadTalkerLinear(src, "talker.text_projection.linear_fc1", cfg.TalkerTextHiddenSize, cfg.TalkerTextHiddenSize, true); err != nil {
		return nil, err
	}
	if m.textProjection2, err = loadTalkerLinear(src, "talker.text_projection.linear_fc2", cfg.TalkerTextHiddenSize, cfg.TalkerHiddenSize, true); err != nil {
		return nil, err
	}
	if m.norm, err = loadTalkerTensor(src, "talker.model.norm.weight", []int{cfg.TalkerHiddenSize}); err != nil {
		return nil, err
	}
	if m.codecHead, err = loadTalkerLinear(src, "talker.codec_head", cfg.TalkerHiddenSize, cfg.TalkerVocabSize, false); err != nil {
		return nil, err
	}
	queryWidth := sizeProduct(cfg.TalkerNumAttentionHeads, cfg.TalkerHeadDim)
	kvWidth := sizeProduct(cfg.TalkerNumKeyValueHeads, cfg.TalkerHeadDim)
	for i := range m.layers {
		prefix := fmt.Sprintf("talker.model.layers.%d", i)
		l := &m.layers[i]
		if l.inputNorm, err = loadTalkerTensor(src, prefix+".input_layernorm.weight", []int{cfg.TalkerHiddenSize}); err != nil {
			return nil, err
		}
		if l.postNorm, err = loadTalkerTensor(src, prefix+".post_attention_layernorm.weight", []int{cfg.TalkerHiddenSize}); err != nil {
			return nil, err
		}
		if l.q, err = loadTalkerLinear(src, prefix+".self_attn.q_proj", cfg.TalkerHiddenSize, queryWidth, false); err != nil {
			return nil, err
		}
		if l.k, err = loadTalkerLinear(src, prefix+".self_attn.k_proj", cfg.TalkerHiddenSize, kvWidth, false); err != nil {
			return nil, err
		}
		if l.v, err = loadTalkerLinear(src, prefix+".self_attn.v_proj", cfg.TalkerHiddenSize, kvWidth, false); err != nil {
			return nil, err
		}
		if l.o, err = loadTalkerLinear(src, prefix+".self_attn.o_proj", queryWidth, cfg.TalkerHiddenSize, false); err != nil {
			return nil, err
		}
		if l.qNorm, err = loadTalkerTensor(src, prefix+".self_attn.q_norm.weight", []int{cfg.TalkerHeadDim}); err != nil {
			return nil, err
		}
		if l.kNorm, err = loadTalkerTensor(src, prefix+".self_attn.k_norm.weight", []int{cfg.TalkerHeadDim}); err != nil {
			return nil, err
		}
		if l.gate, err = loadTalkerLinear(src, prefix+".mlp.gate_proj", cfg.TalkerHiddenSize, cfg.TalkerIntermediateSize, false); err != nil {
			return nil, err
		}
		if l.up, err = loadTalkerLinear(src, prefix+".mlp.up_proj", cfg.TalkerHiddenSize, cfg.TalkerIntermediateSize, false); err != nil {
			return nil, err
		}
		if l.down, err = loadTalkerLinear(src, prefix+".mlp.down_proj", cfg.TalkerIntermediateSize, cfg.TalkerHiddenSize, false); err != nil {
			return nil, err
		}
	}
	return m, nil
}

func loadTalkerLinear(src Float32TensorSource, prefix string, inDim, outDim int, bias bool) (talkerLinear, error) {
	weight, err := loadTalkerTensor(src, prefix+".weight", []int{outDim, inDim})
	if err != nil {
		return talkerLinear{}, err
	}
	l := talkerLinear{weight: weight, inDim: inDim, outDim: outDim}
	if bias {
		l.bias, err = loadTalkerTensor(src, prefix+".bias", []int{outDim})
		if err != nil {
			return talkerLinear{}, err
		}
	}
	return l, nil
}

func loadTalkerTensor(src Float32TensorSource, name string, want []int) ([]float32, error) {
	data, shape, err := src.GetFloat32(name)
	if err != nil {
		return nil, fmt.Errorf("load %s: %w", name, err)
	}
	if !equalTalkerShape(shape, want) {
		return nil, fmt.Errorf("load %s: shape=%v want %v", name, shape, want)
	}
	wantLen := sizeProduct(want...)
	if wantLen < 0 || len(data) != wantLen {
		return nil, fmt.Errorf("load %s: values=%d want %d", name, len(data), wantLen)
	}
	return data, nil
}

func equalTalkerShape(a, b []int) bool {
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

func (l talkerLinear) forward(dst, input []float32) error {
	if len(dst) != l.outDim || len(input) != l.inDim || !simd.GemvRows(dst, input, l.weight, l.outDim, l.inDim) {
		return fmt.Errorf("invalid Qwen3-TTS linear buffers out/in=%d/%d want %d/%d", len(dst), len(input), l.outDim, l.inDim)
	}
	if len(l.bias) != 0 {
		if len(l.bias) != len(dst) {
			return fmt.Errorf("invalid Qwen3-TTS linear bias=%d want %d", len(l.bias), len(dst))
		}
		for i := range dst {
			dst[i] += l.bias[i]
		}
	}
	return nil
}

// ForwardSemantic implements the staged TalkerRuntime boundary. The first
// semantic token depends only on Talker prefill; later tokens also require the
// 15 acoustic embeddings generated by CodePredictor, so this method returns the
// independently executable first token until that stage lands.
func (m *TalkerCPU) ForwardSemantic(plan RuntimeRequestPlan) ([]uint32, error) {
	result, err := m.Prefill(plan)
	if err != nil {
		return nil, err
	}
	return []uint32{result.SemanticToken}, nil
}

// Prefill builds the ten-position CustomVoice prefix, runs causal Talker
// attention, applies reserved-token suppression, and selects the greedy token.
func (m *TalkerCPU) Prefill(plan RuntimeRequestPlan) (TalkerPrefillResult, error) {
	if m == nil {
		return TalkerPrefillResult{}, fmt.Errorf("nil Qwen3-TTS CPU Talker")
	}
	if err := plan.Validate(); err != nil {
		return TalkerPrefillResult{}, err
	}
	if plan.Conditioning.Request.Speaker == "" || plan.Conditioning.Request.Language == "" {
		return TalkerPrefillResult{}, fmt.Errorf("Qwen3-TTS CPU Talker requires CustomVoice speaker and language")
	}
	const prefillTokens = CustomVoiceFirstTextIndex + 1
	if len(plan.Prompt.Text) < prefillTokens || len(plan.Prompt.Codec) != 7 {
		return TalkerPrefillResult{}, fmt.Errorf("invalid Qwen3-TTS CustomVoice prompt lengths text=%d codec=%d", len(plan.Prompt.Text), len(plan.Prompt.Codec))
	}
	languageID, err := plan.Conditioning.Request.Language.TokenID()
	if err != nil {
		return TalkerPrefillResult{}, err
	}
	speakerID, err := plan.Conditioning.Request.Speaker.TokenID()
	if err != nil {
		return TalkerPrefillResult{}, err
	}
	wantTextPrefix := [...]uint32{IMStart, Assistant, Newline, TTSPad, TTSPad, TTSPad, TTSPad, TTSPad, TTSBOS}
	wantCodec := [...]uint32{CodecThink, CodecThinkBOS, languageID, CodecThinkEOS, speakerID, CodecPad, CodecBOS}
	for i, want := range wantTextPrefix {
		if plan.Prompt.Text[i] != want {
			return TalkerPrefillResult{}, fmt.Errorf("invalid Qwen3-TTS CustomVoice text control at %d: got=%d want=%d", i, plan.Prompt.Text[i], want)
		}
	}
	for i, want := range wantCodec {
		if plan.Prompt.Codec[i] != want {
			return TalkerPrefillResult{}, fmt.Errorf("invalid Qwen3-TTS CustomVoice codec control at %d: got=%d want=%d", i, plan.Prompt.Codec[i], want)
		}
	}
	h := m.cfg.TalkerHiddenSize
	inputs := make([]float32, prefillTokens*h)
	for pos := 0; pos < prefillTokens; pos++ {
		projected, err := m.projectText(plan.Prompt.Text[pos])
		if err != nil {
			return TalkerPrefillResult{}, fmt.Errorf("Qwen3-TTS Talker text position %d: %w", pos, err)
		}
		copy(inputs[pos*h:(pos+1)*h], projected)
	}
	for i, id := range plan.Prompt.Codec[:6] {
		if err := addTalkerEmbedding(inputs[(3+i)*h:(4+i)*h], m.codecEmbedding, id, m.cfg.TalkerVocabSize, h); err != nil {
			return TalkerPrefillResult{}, fmt.Errorf("Qwen3-TTS Talker codec position %d: %w", i, err)
		}
	}
	if err := addTalkerEmbedding(inputs[9*h:10*h], m.codecEmbedding, plan.Prompt.Codec[6], m.cfg.TalkerVocabSize, h); err != nil {
		return TalkerPrefillResult{}, fmt.Errorf("Qwen3-TTS Talker codec BOS: %w", err)
	}
	last, err := m.forwardInputs(inputs, prefillTokens)
	if err != nil {
		return TalkerPrefillResult{}, err
	}
	hidden := append([]float32(nil), last...)
	if !simd.RMSNormTo(hidden, m.norm, float32(m.cfg.TalkerRMSNormEps)) {
		return TalkerPrefillResult{}, fmt.Errorf("Qwen3-TTS Talker final RMSNorm failed")
	}
	logits := make([]float32, m.cfg.TalkerVocabSize)
	if err := m.codecHead.forward(logits, hidden); err != nil {
		return TalkerPrefillResult{}, err
	}
	token, err := greedyTalkerToken(logits, CodecEOS)
	if err != nil {
		return TalkerPrefillResult{}, err
	}
	return TalkerPrefillResult{Hidden: hidden, Logits: logits, SemanticToken: token, PrefillTokens: prefillTokens}, nil
}

func (m *TalkerCPU) projectText(id uint32) ([]float32, error) {
	if int(id) >= m.cfg.TalkerTextVocabSize {
		return nil, fmt.Errorf("text token=%d vocab=%d", id, m.cfg.TalkerTextVocabSize)
	}
	start := int(id) * m.cfg.TalkerTextHiddenSize
	embed := m.textEmbedding[start : start+m.cfg.TalkerTextHiddenSize]
	intermediate := make([]float32, m.cfg.TalkerTextHiddenSize)
	if err := m.textProjection1.forward(intermediate, embed); err != nil {
		return nil, err
	}
	if !simd.SiLUTo(intermediate, intermediate) {
		return nil, fmt.Errorf("Qwen3-TTS text projection SiLU failed")
	}
	out := make([]float32, m.cfg.TalkerHiddenSize)
	if err := m.textProjection2.forward(out, intermediate); err != nil {
		return nil, err
	}
	return out, nil
}

func addTalkerEmbedding(dst, table []float32, id uint32, vocab, width int) error {
	if int(id) >= vocab || len(dst) != width {
		return fmt.Errorf("embedding token=%d vocab=%d width=%d dst=%d", id, vocab, width, len(dst))
	}
	start := int(id) * width
	if start < 0 || start+width > len(table) {
		return fmt.Errorf("embedding token=%d exceeds table values=%d", id, len(table))
	}
	for i := range dst {
		dst[i] += table[start+i]
	}
	return nil
}

func (m *TalkerCPU) forwardInputs(inputs []float32, seqLen int) ([]float32, error) {
	h := m.cfg.TalkerHiddenSize
	if seqLen <= 0 || len(inputs) != seqLen*h {
		return nil, fmt.Errorf("invalid Qwen3-TTS Talker input values=%d sequence=%d hidden=%d", len(inputs), seqLen, h)
	}
	kvWidth := m.cfg.TalkerNumKeyValueHeads * m.cfg.TalkerHeadDim
	kCaches := make([][]float32, len(m.layers))
	vCaches := make([][]float32, len(m.layers))
	for i := range m.layers {
		kCaches[i] = make([]float32, 0, seqLen*kvWidth)
		vCaches[i] = make([]float32, 0, seqLen*kvWidth)
	}
	// Qwen3-TTS supplies identical temporal, height, and width position IDs.
	// Its interleaved MRoPE therefore reduces exactly to standard full RoPE.
	rope := simd.BuildRoPEFreqs(seqLen, m.cfg.TalkerHeadDim/2, m.cfg.TalkerHeadDim, m.cfg.TalkerRoPETheta)
	if len(rope) == 0 {
		return nil, fmt.Errorf("Qwen3-TTS Talker RoPE table is empty")
	}
	var current []float32
	for pos := 0; pos < seqLen; pos++ {
		current = append(current[:0], inputs[pos*h:(pos+1)*h]...)
		for layerIndex := range m.layers {
			var err error
			current, kCaches[layerIndex], vCaches[layerIndex], err = m.layers[layerIndex].forward(current, kCaches[layerIndex], vCaches[layerIndex], pos, rope, m.cfg)
			if err != nil {
				return nil, fmt.Errorf("Qwen3-TTS Talker layer %d position %d: %w", layerIndex, pos, err)
			}
		}
	}
	return current, nil
}

func (l *talkerCPULayer) forward(input, pastK, pastV []float32, pos int, rope []float32, cfg ParsedConfig) ([]float32, []float32, []float32, error) {
	h := cfg.TalkerHiddenSize
	queryWidth := cfg.TalkerNumAttentionHeads * cfg.TalkerHeadDim
	kvWidth := cfg.TalkerNumKeyValueHeads * cfg.TalkerHeadDim
	if len(input) != h {
		return nil, nil, nil, fmt.Errorf("input=%d want %d", len(input), h)
	}
	normed := append([]float32(nil), input...)
	if !simd.RMSNormTo(normed, l.inputNorm, float32(cfg.TalkerRMSNormEps)) {
		return nil, nil, nil, fmt.Errorf("input RMSNorm failed")
	}
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
	if err := normTalkerHeads(q, l.qNorm, cfg.TalkerNumAttentionHeads, cfg.TalkerHeadDim, float32(cfg.TalkerRMSNormEps)); err != nil {
		return nil, nil, nil, err
	}
	if err := normTalkerHeads(k, l.kNorm, cfg.TalkerNumKeyValueHeads, cfg.TalkerHeadDim, float32(cfg.TalkerRMSNormEps)); err != nil {
		return nil, nil, nil, err
	}
	if !simd.ApplyRoPETo(q, rope, pos, cfg.TalkerNumAttentionHeads, cfg.TalkerHeadDim) || !simd.ApplyRoPETo(k, rope, pos, cfg.TalkerNumKeyValueHeads, cfg.TalkerHeadDim) {
		return nil, nil, nil, fmt.Errorf("RoPE failed")
	}
	kAll := append(pastK, k...)
	vAll := append(pastV, v...)
	attention := make([]float32, queryWidth)
	scores := make([]float32, pos+1)
	scale := float32(1 / math.Sqrt(float64(cfg.TalkerHeadDim)))
	if !simd.GQAAttentionScaleTo(attention, scores, q, kAll, vAll, pos+1, cfg.TalkerNumAttentionHeads, cfg.TalkerNumKeyValueHeads, cfg.TalkerHeadDim, scale) {
		return nil, nil, nil, fmt.Errorf("GQA attention failed")
	}
	projected := make([]float32, h)
	if err := l.o.forward(projected, attention); err != nil {
		return nil, nil, nil, err
	}
	residual := make([]float32, h)
	if !simd.VecAddTo(residual, input, projected) {
		return nil, nil, nil, fmt.Errorf("attention residual failed")
	}
	mlpInput := append([]float32(nil), residual...)
	if !simd.RMSNormTo(mlpInput, l.postNorm, float32(cfg.TalkerRMSNormEps)) {
		return nil, nil, nil, fmt.Errorf("post-attention RMSNorm failed")
	}
	gate := make([]float32, cfg.TalkerIntermediateSize)
	up := make([]float32, cfg.TalkerIntermediateSize)
	if err := l.gate.forward(gate, mlpInput); err != nil {
		return nil, nil, nil, err
	}
	if err := l.up.forward(up, mlpInput); err != nil {
		return nil, nil, nil, err
	}
	if !simd.SiLUMulTo(gate, gate, up) {
		return nil, nil, nil, fmt.Errorf("MLP SwiGLU failed")
	}
	down := make([]float32, h)
	if err := l.down.forward(down, gate); err != nil {
		return nil, nil, nil, err
	}
	out := make([]float32, h)
	if !simd.VecAddTo(out, residual, down) {
		return nil, nil, nil, fmt.Errorf("MLP residual failed")
	}
	return out, kAll, vAll, nil
}

func normTalkerHeads(x, weight []float32, heads, headDim int, eps float32) error {
	if heads <= 0 || headDim <= 0 || len(x) != heads*headDim || len(weight) != headDim {
		return fmt.Errorf("invalid Qwen3-TTS QK norm dims values=%d weight=%d heads=%d head_dim=%d", len(x), len(weight), heads, headDim)
	}
	for head := 0; head < heads; head++ {
		if !simd.RMSNormTo(x[head*headDim:(head+1)*headDim], weight, eps) {
			return fmt.Errorf("Qwen3-TTS QK norm head %d failed", head)
		}
	}
	return nil
}

func greedyTalkerToken(logits []float32, eos uint32) (uint32, error) {
	if len(logits) <= 1024 || int(eos) >= len(logits) {
		return 0, fmt.Errorf("invalid Qwen3-TTS Talker logits=%d eos=%d", len(logits), eos)
	}
	suppressStart := len(logits) - 1024
	best := -1
	bestValue := float32(math.Inf(-1))
	for i, value := range logits {
		if i >= suppressStart && uint32(i) != eos {
			continue
		}
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 1) {
			return 0, fmt.Errorf("non-finite Qwen3-TTS Talker logit at %d", i)
		}
		if best < 0 || value > bestValue {
			best, bestValue = i, value
		}
	}
	if best < 0 || math.IsInf(float64(bestValue), -1) {
		return 0, fmt.Errorf("Qwen3-TTS Talker has no finite unsuppressed logit")
	}
	return uint32(best), nil
}
