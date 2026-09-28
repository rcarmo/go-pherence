package lfm2

import (
	"fmt"
	"math"
	"sort"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

type lfm2ExpertCPU struct {
	gateUp []float32
	down   []float32
}

// MoECPU owns one routed LFM2 feed-forward block, including the non-trainable
// expert-selection bias when enabled by the checkpoint.
type MoECPU struct {
	cfg        Config
	layer      int
	router     lfm2Linear
	expertBias []float32
	experts    []lfm2ExpertCPU
}

type RouterSelection struct {
	ExpertIDs []int
	Weights   []float32
}

func LoadMoECPU(src Float32TensorSource, cfg Config, layer int) (*MoECPU, error) {
	if src == nil {
		return nil, fmt.Errorf("nil LFM2 MoE tensor source")
	}
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if layer < cfg.NumDenseLayers || layer < 0 || layer >= cfg.NumHiddenLayers {
		return nil, fmt.Errorf("LFM2 layer %d is not a routed MoE layer", layer)
	}
	prefix := fmt.Sprintf("model.layers.%d.feed_forward", layer)
	m := &MoECPU{cfg: cfg, layer: layer, experts: make([]lfm2ExpertCPU, cfg.NumExperts)}
	var err error
	if m.router, err = loadLFM2Linear(src, prefix+".gate", cfg.HiddenSize, cfg.NumExperts, false); err != nil {
		return nil, err
	}
	if cfg.UseExpertBias {
		if m.expertBias, err = loadLFM2Tensor(src, prefix+".expert_bias", []int{cfg.NumExperts}); err != nil {
			return nil, err
		}
	}
	gateUpWidth := sizeProduct(2, cfg.MoEIntermediateSize)
	gateUp, err := loadLFM2Tensor(src, prefix+".experts.gate_up_proj", []int{cfg.NumExperts, gateUpWidth, cfg.HiddenSize})
	if err != nil {
		return nil, err
	}
	down, err := loadLFM2Tensor(src, prefix+".experts.down_proj", []int{cfg.NumExperts, cfg.HiddenSize, cfg.MoEIntermediateSize})
	if err != nil {
		return nil, err
	}
	gateUpPerExpert := sizeProduct(gateUpWidth, cfg.HiddenSize)
	downPerExpert := sizeProduct(cfg.HiddenSize, cfg.MoEIntermediateSize)
	for expert := range m.experts {
		m.experts[expert] = lfm2ExpertCPU{
			gateUp: gateUp[expert*gateUpPerExpert : (expert+1)*gateUpPerExpert],
			down:   down[expert*downPerExpert : (expert+1)*downPerExpert],
		}
	}
	return m, nil
}

// Route applies sigmoid routing, optional selection-only expert bias, stable
// descending top-k selection, selected raw-weight normalization, and routed
// scaling exactly in that order.
func (m *MoECPU) Route(input []float32) (RouterSelection, error) {
	if m == nil {
		return RouterSelection{}, fmt.Errorf("nil LFM2 CPU MoE")
	}
	if len(input) != m.cfg.HiddenSize {
		return RouterSelection{}, fmt.Errorf("invalid LFM2 router input=%d want %d", len(input), m.cfg.HiddenSize)
	}
	logits := make([]float32, m.cfg.NumExperts)
	if err := m.router.forward(logits, input); err != nil {
		return RouterSelection{}, err
	}
	type candidate struct {
		id       int
		weight   float32
		priority float32
	}
	candidates := make([]candidate, len(logits))
	for i, logit := range logits {
		if math.IsNaN(float64(logit)) || math.IsInf(float64(logit), 0) {
			return RouterSelection{}, fmt.Errorf("non-finite LFM2 router logit at %d", i)
		}
		weight := float32(1 / (1 + math.Exp(float64(-logit))))
		priority := weight
		if len(m.expertBias) != 0 {
			priority += m.expertBias[i]
		}
		candidates[i] = candidate{id: i, weight: weight, priority: priority}
	}
	sort.SliceStable(candidates, func(i, j int) bool {
		if candidates[i].priority == candidates[j].priority {
			return candidates[i].id < candidates[j].id
		}
		return candidates[i].priority > candidates[j].priority
	})
	selected := candidates[:m.cfg.NumExpertsPerTok]
	out := RouterSelection{ExpertIDs: make([]int, len(selected)), Weights: make([]float32, len(selected))}
	var sum float32
	for i, item := range selected {
		out.ExpertIDs[i], out.Weights[i] = item.id, item.weight
		sum += item.weight
	}
	if m.cfg.NormTopKProb {
		sum += 1e-6
		for i := range out.Weights {
			out.Weights[i] /= sum
		}
	}
	for i := range out.Weights {
		out.Weights[i] *= float32(m.cfg.RoutedScalingFactor)
	}
	return out, nil
}

// ForwardToken routes one hidden vector and sums the selected SwiGLU experts.
func (m *MoECPU) ForwardToken(input []float32) ([]float32, RouterSelection, error) {
	selection, err := m.Route(input)
	if err != nil {
		return nil, RouterSelection{}, err
	}
	h, intermediate := m.cfg.HiddenSize, m.cfg.MoEIntermediateSize
	out := make([]float32, h)
	for selected, expertID := range selection.ExpertIDs {
		expert := m.experts[expertID]
		gateUp := make([]float32, 2*intermediate)
		if !simd.GemvRows(gateUp, input, expert.gateUp, 2*intermediate, h) {
			return nil, RouterSelection{}, fmt.Errorf("LFM2 expert %d gate/up projection failed", expertID)
		}
		gate, up := gateUp[:intermediate], gateUp[intermediate:]
		if !simd.SiLUMulTo(gate, gate, up) {
			return nil, RouterSelection{}, fmt.Errorf("LFM2 expert %d SwiGLU failed", expertID)
		}
		down := make([]float32, h)
		if !simd.GemvRows(down, gate, expert.down, h, intermediate) {
			return nil, RouterSelection{}, fmt.Errorf("LFM2 expert %d down projection failed", expertID)
		}
		for i := range out {
			out[i] += selection.Weights[selected] * down[i]
		}
	}
	return out, selection, nil
}
