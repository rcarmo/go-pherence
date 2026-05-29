package lfm2

import "fmt"

type FFNKind string

const (
	FFNDense FFNKind = "dense"
	FFNMoE   FFNKind = "moe"
)

type LayerExecutionStep struct {
	Index int       `json:"index"`
	Kind  LayerKind `json:"kind"`
	FFN   FFNKind   `json:"ffn"`
}

type ExecutionPlan struct {
	Steps        []LayerExecutionStep `json:"steps"`
	DenseIndices []int                `json:"dense_indices"`
	MoEIndices   []int                `json:"moe_indices"`
}

func NewExecutionPlan(cfg Config) (ExecutionPlan, error) {
	schedule, err := NewLayerSchedule(cfg)
	if err != nil {
		return ExecutionPlan{}, err
	}
	if cfg.NumDenseLayers < 0 || cfg.NumDenseLayers > cfg.NumHiddenLayers {
		return ExecutionPlan{}, fmt.Errorf("invalid LFM2 dense layer count=%d layers=%d", cfg.NumDenseLayers, cfg.NumHiddenLayers)
	}
	plan := ExecutionPlan{Steps: make([]LayerExecutionStep, 0, len(schedule.Steps))}
	for _, step := range schedule.Steps {
		exec := LayerExecutionStep{Index: step.Index, Kind: step.Kind, FFN: FFNMoE}
		if step.Index < cfg.NumDenseLayers {
			exec.FFN = FFNDense
			plan.DenseIndices = append(plan.DenseIndices, step.Index)
		} else {
			plan.MoEIndices = append(plan.MoEIndices, step.Index)
		}
		plan.Steps = append(plan.Steps, exec)
	}
	if err := plan.Validate(cfg.NumHiddenLayers); err != nil {
		return ExecutionPlan{}, err
	}
	return plan, nil
}

func (p ExecutionPlan) Validate(numLayers int) error {
	if numLayers <= 0 || len(p.Steps) != numLayers {
		return fmt.Errorf("invalid LFM2 execution plan length=%d want=%d", len(p.Steps), numLayers)
	}
	dense, moe := 0, 0
	for pos, step := range p.Steps {
		if step.Index != pos {
			return fmt.Errorf("invalid LFM2 execution position=%d index=%d", pos, step.Index)
		}
		switch step.Kind {
		case LayerConv, LayerFullAttention:
		default:
			return fmt.Errorf("invalid LFM2 execution layer kind at %d: %q", step.Index, step.Kind)
		}
		switch step.FFN {
		case FFNDense:
			dense++
		case FFNMoE:
			moe++
		default:
			return fmt.Errorf("invalid LFM2 execution FFN kind at %d: %q", step.Index, step.FFN)
		}
	}
	if dense != len(p.DenseIndices) || moe != len(p.MoEIndices) {
		return fmt.Errorf("invalid LFM2 execution index counts: dense=%d/%d moe=%d/%d", dense, len(p.DenseIndices), moe, len(p.MoEIndices))
	}
	return nil
}

func (p ExecutionPlan) IsDenseLayer(layer int) bool {
	for _, idx := range p.DenseIndices {
		if idx == layer {
			return true
		}
	}
	return false
}

func (p ExecutionPlan) IsMoELayer(layer int) bool {
	for _, idx := range p.MoEIndices {
		if idx == layer {
			return true
		}
	}
	return false
}
