package lfm2

import (
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/loader/weights"
)

type denseFFNCPU struct {
	gate, up, down lfm2Linear
}

func loadDenseFFNCPU(src Float32TensorSource, cfg Config, layer int) (*denseFFNCPU, error) {
	if layer < 0 || layer >= cfg.NumDenseLayers {
		return nil, fmt.Errorf("LFM2 layer %d is not a dense FFN layer", layer)
	}
	prefix := fmt.Sprintf("model.layers.%d.feed_forward", layer)
	m := &denseFFNCPU{}
	var err error
	if m.gate, err = loadLFM2Linear(src, prefix+".w1", cfg.HiddenSize, cfg.IntermediateSize, false); err != nil {
		return nil, err
	}
	if m.up, err = loadLFM2Linear(src, prefix+".w3", cfg.HiddenSize, cfg.IntermediateSize, false); err != nil {
		return nil, err
	}
	if m.down, err = loadLFM2Linear(src, prefix+".w2", cfg.IntermediateSize, cfg.HiddenSize, false); err != nil {
		return nil, err
	}
	return m, nil
}

func (m *denseFFNCPU) forward(input []float32) ([]float32, error) {
	if m == nil {
		return nil, fmt.Errorf("nil LFM2 dense FFN")
	}
	gate := make([]float32, m.gate.outDim)
	up := make([]float32, m.up.outDim)
	if err := m.gate.forward(gate, input); err != nil {
		return nil, err
	}
	if err := m.up.forward(up, input); err != nil {
		return nil, err
	}
	if !simd.SiLUMulTo(gate, gate, up) {
		return nil, fmt.Errorf("LFM2 dense SwiGLU failed")
	}
	out := make([]float32, m.down.outDim)
	if err := m.down.forward(out, gate); err != nil {
		return nil, err
	}
	return out, nil
}

type decoderCPULayer struct {
	operatorNorm []float32
	ffnNorm      []float32
	conv         *ShortConvCPU
	attention    *FullAttentionCPU
	dense        *denseFFNCPU
	moe          *MoECPU
}

// DecoderCPUState owns every recurrent convolution state and full-attention KV
// cache. Slots are indexed by physical decoder layer; unused slots are empty.
type DecoderCPUState struct {
	Conv      [][]float32
	Attention []AttentionCPUState
	Pos       int
}

func (s DecoderCPUState) Clone() DecoderCPUState {
	out := DecoderCPUState{Conv: make([][]float32, len(s.Conv)), Attention: make([]AttentionCPUState, len(s.Attention)), Pos: s.Pos}
	for i := range s.Conv {
		out.Conv[i] = append([]float32(nil), s.Conv[i]...)
	}
	for i := range s.Attention {
		out.Attention[i] = s.Attention[i].Clone()
	}
	return out
}

// DecoderCPU composes the LFM2 embedding, operator, FFN, and output stages. All
// checkpoint tensors are owned F32 data and all request state is caller-local.
type DecoderCPU struct {
	cfg       Config
	embedding *EmbeddingCPU
	layers    []decoderCPULayer
}

func LoadDecoderCPUFromDir(dir string, cfg Config) (*DecoderCPU, error) {
	src, err := weights.OpenSafetensors(dir)
	if err != nil {
		return nil, err
	}
	defer src.Close()
	return LoadDecoderCPU(src, cfg)
}

func LoadDecoderCPU(src Float32TensorSource, cfg Config) (*DecoderCPU, error) {
	if src == nil {
		return nil, fmt.Errorf("nil LFM2 decoder tensor source")
	}
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	embedding, err := LoadEmbeddingCPU(src, cfg)
	if err != nil {
		return nil, err
	}
	m := &DecoderCPU{cfg: cfg, embedding: embedding, layers: make([]decoderCPULayer, cfg.NumHiddenLayers)}
	for layer := range m.layers {
		prefix := fmt.Sprintf("model.layers.%d", layer)
		l := &m.layers[layer]
		if l.operatorNorm, err = loadLFM2Tensor(src, prefix+".operator_norm.weight", []int{cfg.HiddenSize}); err != nil {
			return nil, err
		}
		if l.ffnNorm, err = loadLFM2Tensor(src, prefix+".ffn_norm.weight", []int{cfg.HiddenSize}); err != nil {
			return nil, err
		}
		switch cfg.LayerTypes[layer] {
		case "conv":
			if l.conv, err = LoadShortConvCPU(src, cfg, layer); err != nil {
				return nil, err
			}
		case "full_attention":
			if l.attention, err = LoadFullAttentionCPU(src, cfg, layer); err != nil {
				return nil, err
			}
		}
		if layer < cfg.NumDenseLayers {
			if l.dense, err = loadDenseFFNCPU(src, cfg, layer); err != nil {
				return nil, err
			}
		} else if l.moe, err = LoadMoECPU(src, cfg, layer); err != nil {
			return nil, err
		}
	}
	return m, nil
}

func (m *DecoderCPU) NewState() DecoderCPUState {
	if m == nil {
		return DecoderCPUState{}
	}
	state := DecoderCPUState{Conv: make([][]float32, len(m.layers)), Attention: make([]AttentionCPUState, len(m.layers))}
	for layer := range m.layers {
		if m.layers[layer].conv != nil {
			state.Conv[layer] = m.layers[layer].conv.NewState()
		}
	}
	return state
}

// ForwardToken executes one token embedding through all decoder layers. It
// clones the caller's state and commits each layer update only to that clone.
func (m *DecoderCPU) ForwardToken(token uint32, state DecoderCPUState) ([]float32, DecoderCPUState, error) {
	if m == nil || m.embedding == nil {
		return nil, state, fmt.Errorf("nil LFM2 CPU decoder")
	}
	if err := m.embedding.cfg.Validate(); err != nil {
		return nil, state, err
	}
	if int(token) >= m.cfg.VocabSize {
		return nil, state, fmt.Errorf("invalid LFM2 token id=%d vocab=%d", token, m.cfg.VocabSize)
	}
	maxPosition := m.cfg.MaxPositionEmbeddings
	if maxPosition == 0 {
		maxPosition = 128000
	}
	if state.Pos < 0 || state.Pos >= maxPosition || len(state.Conv) != len(m.layers) || len(state.Attention) != len(m.layers) {
		return nil, state, fmt.Errorf("invalid LFM2 decoder state position/layers=%d/%d/%d", state.Pos, len(state.Conv), len(state.Attention))
	}
	next := state.Clone()
	start := int(token) * m.cfg.HiddenSize
	hidden := append([]float32(nil), m.embedding.embedding[start:start+m.cfg.HiddenSize]...)
	eps := float32(m.cfg.NormEps)
	if eps == 0 {
		eps = 1e-5
	}
	for layer := range m.layers {
		l := &m.layers[layer]
		operatorInput := append([]float32(nil), hidden...)
		if !simd.RMSNormTo(operatorInput, l.operatorNorm, eps) {
			return nil, state, fmt.Errorf("LFM2 layer %d operator RMSNorm failed", layer)
		}
		var operatorOut []float32
		var err error
		if l.conv != nil {
			operatorOut, next.Conv[layer], err = l.conv.ForwardToken(operatorInput, next.Conv[layer])
		} else {
			operatorOut, next.Attention[layer], err = l.attention.ForwardToken(operatorInput, next.Attention[layer], state.Pos)
		}
		if err != nil {
			return nil, state, fmt.Errorf("LFM2 layer %d operator: %w", layer, err)
		}
		if !simd.VecAddTo(hidden, hidden, operatorOut) {
			return nil, state, fmt.Errorf("LFM2 layer %d operator residual failed", layer)
		}
		ffnInput := append([]float32(nil), hidden...)
		if !simd.RMSNormTo(ffnInput, l.ffnNorm, eps) {
			return nil, state, fmt.Errorf("LFM2 layer %d FFN RMSNorm failed", layer)
		}
		var ffnOut []float32
		if l.dense != nil {
			ffnOut, err = l.dense.forward(ffnInput)
		} else {
			ffnOut, _, err = l.moe.ForwardToken(ffnInput)
		}
		if err != nil {
			return nil, state, fmt.Errorf("LFM2 layer %d FFN: %w", layer, err)
		}
		if !simd.VecAddTo(hidden, hidden, ffnOut) {
			return nil, state, fmt.Errorf("LFM2 layer %d FFN residual failed", layer)
		}
	}
	next.Pos++
	return hidden, next, nil
}

// Generate performs deterministic greedy decoding for the validated request.
func (m *DecoderCPU) Generate(plan RuntimeRequestPlan) ([]uint32, error) {
	if m == nil {
		return nil, fmt.Errorf("nil LFM2 CPU decoder")
	}
	if err := plan.Validate(); err != nil {
		return nil, err
	}
	state := m.NewState()
	var hidden []float32
	var err error
	for _, token := range plan.Tokens {
		hidden, state, err = m.ForwardToken(token, state)
		if err != nil {
			return nil, err
		}
	}
	generated := make([]uint32, 0, plan.MaxNewTokens)
	for len(generated) < plan.MaxNewTokens {
		logits, err := m.embedding.FinalLogits(hidden)
		if err != nil {
			return nil, err
		}
		token, err := greedyLFM2Token(logits)
		if err != nil {
			return nil, err
		}
		generated = append(generated, token)
		if len(generated) == plan.MaxNewTokens {
			break
		}
		hidden, state, err = m.ForwardToken(token, state)
		if err != nil {
			return nil, err
		}
	}
	return generated, nil
}

func greedyLFM2Token(logits []float32) (uint32, error) {
	if len(logits) == 0 {
		return 0, fmt.Errorf("empty LFM2 logits")
	}
	best := -1
	bestValue := float32(math.Inf(-1))
	for i, value := range logits {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 1) {
			return 0, fmt.Errorf("non-finite LFM2 logit at %d", i)
		}
		if best < 0 || value > bestValue {
			best, bestValue = i, value
		}
	}
	if best < 0 || math.IsInf(float64(bestValue), -1) {
		return 0, fmt.Errorf("LFM2 logits have no finite value")
	}
	return uint32(best), nil
}
