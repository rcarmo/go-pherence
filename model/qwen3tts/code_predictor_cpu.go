package qwen3tts

import (
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/loader/weights"
)

// CodePredictorCPU owns the 0.6B CustomVoice predictor weights. Each call
// creates fresh KV state for its 2-token prefill and 14 subsequent steps.
// It does not perform Talker continuation or waveform decoding.
type CodePredictorCPU struct {
	cfg        ParsedConfig
	embeddings [][]float32
	heads      []talkerLinear
	layers     []talkerCPULayer
	norm       []float32
}

// LoadCodePredictorCPUFromDir copies the predictor tensors before closing the
// safetensors source. The released-model test hashes the checkpoint first.
func LoadCodePredictorCPUFromDir(dir string, cfg ParsedConfig) (*CodePredictorCPU, error) {
	src, err := weights.OpenSafetensors(dir)
	if err != nil {
		return nil, err
	}
	defer src.Close()
	return LoadCodePredictorCPU(src, cfg)
}

// LoadCodePredictorCPU binds only the equal-width 0.6B CustomVoice predictor.
// The wider 1.7B small_to_mtp_projection requires a separate implementation.
func LoadCodePredictorCPU(src Float32TensorSource, cfg ParsedConfig) (*CodePredictorCPU, error) {
	if src == nil {
		return nil, fmt.Errorf("nil Qwen3-TTS CodePredictor tensor source")
	}
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if cfg.ModelType != CustomVoice || cfg.ModelSize == "1b7" || cfg.CPNumCodeGroups != 16 || cfg.CPHiddenSize != cfg.TalkerHiddenSize || cfg.CPHeadDim%2 != 0 {
		return nil, fmt.Errorf("unsupported Qwen3-TTS CodePredictor geometry")
	}
	m := &CodePredictorCPU{cfg: cfg, embeddings: make([][]float32, 15), heads: make([]talkerLinear, 15), layers: make([]talkerCPULayer, cfg.CPNumHiddenLayers)}
	const prefix = "talker.code_predictor."
	var err error
	for i := range m.embeddings {
		m.embeddings[i], err = loadTalkerTensor(src, fmt.Sprintf("%smodel.codec_embedding.%d.weight", prefix, i), []int{cfg.CPVocabSize, cfg.CPHiddenSize})
		if err != nil {
			return nil, err
		}
		m.heads[i], err = loadTalkerLinear(src, fmt.Sprintf("%slm_head.%d", prefix, i), cfg.CPHiddenSize, cfg.CPVocabSize, false)
		if err != nil {
			return nil, err
		}
	}
	m.norm, err = loadTalkerTensor(src, prefix+"model.norm.weight", []int{cfg.CPHiddenSize})
	if err != nil {
		return nil, err
	}
	queryWidth := sizeProduct(cfg.CPNumAttentionHeads, cfg.CPHeadDim)
	kvWidth := sizeProduct(cfg.CPNumKeyValueHeads, cfg.CPHeadDim)
	for i := range m.layers {
		base := fmt.Sprintf("%smodel.layers.%d", prefix, i)
		l := &m.layers[i]
		if l.inputNorm, err = loadTalkerTensor(src, base+".input_layernorm.weight", []int{cfg.CPHiddenSize}); err != nil {
			return nil, err
		}
		if l.postNorm, err = loadTalkerTensor(src, base+".post_attention_layernorm.weight", []int{cfg.CPHiddenSize}); err != nil {
			return nil, err
		}
		if l.q, err = loadTalkerLinear(src, base+".self_attn.q_proj", cfg.CPHiddenSize, queryWidth, false); err != nil {
			return nil, err
		}
		if l.k, err = loadTalkerLinear(src, base+".self_attn.k_proj", cfg.CPHiddenSize, kvWidth, false); err != nil {
			return nil, err
		}
		if l.v, err = loadTalkerLinear(src, base+".self_attn.v_proj", cfg.CPHiddenSize, kvWidth, false); err != nil {
			return nil, err
		}
		if l.o, err = loadTalkerLinear(src, base+".self_attn.o_proj", queryWidth, cfg.CPHiddenSize, false); err != nil {
			return nil, err
		}
		if l.qNorm, err = loadTalkerTensor(src, base+".self_attn.q_norm.weight", []int{cfg.CPHeadDim}); err != nil {
			return nil, err
		}
		if l.kNorm, err = loadTalkerTensor(src, base+".self_attn.k_norm.weight", []int{cfg.CPHeadDim}); err != nil {
			return nil, err
		}
		if l.gate, err = loadTalkerLinear(src, base+".mlp.gate_proj", cfg.CPHiddenSize, cfg.CPIntermediateSize, false); err != nil {
			return nil, err
		}
		if l.up, err = loadTalkerLinear(src, base+".mlp.up_proj", cfg.CPHiddenSize, cfg.CPIntermediateSize, false); err != nil {
			return nil, err
		}
		if l.down, err = loadTalkerLinear(src, base+".mlp.down_proj", cfg.CPIntermediateSize, cfg.CPHiddenSize, false); err != nil {
			return nil, err
		}
	}
	return m, nil
}

// FirstAcousticFrame computes the 15 acoustic codes conditioned on the
// Talker's normed final hidden row and chosen semantic token. firstLogits is
// the raw 2048-wide first head before greedy selection. Results are owned.
func (m *CodePredictorCPU) FirstAcousticFrame(talker *TalkerCPU, hidden []float32, semantic uint32) (codes []uint32, firstLogits []float32, err error) {
	return m.firstAcousticFrameWithWorkspace(talker, hidden, semantic, nil)
}

// codePredictorWorkspace is owned by a single request. Each frame resets KV
// lengths, while retaining bounded capacity and per-layer scratch. No returned
// codes or logits alias it.
type codePredictorWorkspace struct {
	layers                []talkerLayerScratch
	kCache, vCache        [][]float32
	rope                  []float32
	input, normed, logits []float32
}

func newCodePredictorWorkspace(cfg ParsedConfig, layers int) *codePredictorWorkspace {
	cpCfg := codePredictorTalkerConfig(cfg)
	const seqLen = 16
	kvWidth := cfg.CPNumKeyValueHeads * cfg.CPHeadDim
	w := &codePredictorWorkspace{
		layers: make([]talkerLayerScratch, layers), kCache: make([][]float32, layers), vCache: make([][]float32, layers),
		rope:  simd.BuildRoPEFreqs(seqLen, cfg.CPHeadDim/2, cfg.CPHeadDim, cfg.CPRoPETheta),
		input: make([]float32, cfg.CPHiddenSize), normed: make([]float32, cfg.CPHiddenSize), logits: make([]float32, cfg.CPVocabSize),
	}
	for i := range w.layers {
		w.layers[i] = newTalkerLayerScratch(cpCfg, seqLen)
		w.kCache[i], w.vCache[i] = make([]float32, 0, seqLen*kvWidth), make([]float32, 0, seqLen*kvWidth)
	}
	return w
}

func codePredictorTalkerConfig(cfg ParsedConfig) ParsedConfig {
	cfg.TalkerHiddenSize, cfg.TalkerIntermediateSize = cfg.CPHiddenSize, cfg.CPIntermediateSize
	cfg.TalkerNumAttentionHeads, cfg.TalkerNumKeyValueHeads, cfg.TalkerHeadDim = cfg.CPNumAttentionHeads, cfg.CPNumKeyValueHeads, cfg.CPHeadDim
	cfg.TalkerRMSNormEps, cfg.TalkerRoPETheta = cfg.CPRMSNormEps, cfg.CPRoPETheta
	return cfg
}

func (m *CodePredictorCPU) firstAcousticFrameWithWorkspace(talker *TalkerCPU, hidden []float32, semantic uint32, w *codePredictorWorkspace) (codes []uint32, firstLogits []float32, err error) {
	if m == nil || talker == nil || m.cfg.CPHiddenSize != talker.cfg.TalkerHiddenSize || len(hidden) != m.cfg.CPHiddenSize || int(semantic) >= talker.cfg.TalkerVocabSize {
		return nil, nil, fmt.Errorf("invalid Qwen3-TTS CodePredictor input")
	}
	for _, v := range hidden {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return nil, nil, fmt.Errorf("non-finite Qwen3-TTS CodePredictor hidden")
		}
	}
	cpCfg := codePredictorTalkerConfig(m.cfg)
	const seqLen = 16 // positions 0,1 prefill; positions 2..15 decode
	if w == nil {
		w = newCodePredictorWorkspace(m.cfg, len(m.layers))
	}
	if len(w.rope) == 0 {
		return nil, nil, fmt.Errorf("Qwen3-TTS CodePredictor RoPE unavailable")
	}
	for i := range w.kCache {
		w.kCache[i], w.vCache[i] = w.kCache[i][:0], w.vCache[i][:0]
	}
	input := w.input
	copy(input, hidden)
	semanticEmbed := talker.codecEmbedding[int(semantic)*m.cfg.CPHiddenSize : (int(semantic)+1)*m.cfg.CPHiddenSize]
	codes = make([]uint32, 15)
	for pos := 0; pos < seqLen; pos++ {
		if pos == 1 {
			copy(input, semanticEmbed)
		}
		if pos >= 2 {
			id := codes[pos-2]
			if int(id) >= m.cfg.CPVocabSize {
				return nil, nil, fmt.Errorf("invalid CodePredictor prior code %d", id)
			}
			copy(input, m.embeddings[pos-2][int(id)*m.cfg.CPHiddenSize:(int(id)+1)*m.cfg.CPHiddenSize])
		}
		for i := range m.layers {
			input, w.kCache[i], w.vCache[i], err = m.layers[i].forwardWithScratch(input, w.kCache[i], w.vCache[i], pos, w.rope, cpCfg, &w.layers[i])
			if err != nil {
				return nil, nil, fmt.Errorf("Qwen3-TTS CodePredictor layer %d position %d: %w", i, pos, err)
			}
		}
		if pos == 0 {
			continue
		}
		copy(w.normed, input)
		if !simd.RMSNormTo(w.normed, m.norm, float32(m.cfg.CPRMSNormEps)) {
			return nil, nil, fmt.Errorf("Qwen3-TTS CodePredictor final norm failed")
		}
		if err = m.heads[pos-1].forward(w.logits, w.normed); err != nil {
			return nil, nil, err
		}
		best := 0
		for j, v := range w.logits {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return nil, nil, fmt.Errorf("non-finite Qwen3-TTS CodePredictor logit")
			}
			if v > w.logits[best] {
				best = j
			}
		}
		codes[pos-1] = uint32(best)
		if pos == 1 {
			firstLogits = append([]float32(nil), w.logits...)
		}
	}
	return codes, firstLogits, nil
}
