package lfm2

import (
	"encoding/json"
	"fmt"
	"os"
)

// ReferenceMetadata is a small, commit-safe LFM2 fixture used before runtime
// parity tensors are available. It captures the published config plus optional
// tensor coverage counts from a local safetensors header.
type ReferenceMetadata struct {
	Name    string         `json:"name,omitempty"`
	Config  Config         `json:"config"`
	Tensors TensorCoverage `json:"tensors,omitempty"`
}

type ReferenceCoverage struct {
	ConfigMetadata       bool     `json:"config_metadata"`
	RuntimePlan          bool     `json:"runtime_plan"`
	TensorCoverage       bool     `json:"tensor_coverage"`
	TensorReadiness      bool     `json:"tensor_readiness"`
	TokenizationFixture  bool     `json:"tokenization_fixture"`
	FirstTokenLogits     bool     `json:"first_token_logits"`
	ConvLayerReference   bool     `json:"conv_layer_reference"`
	AttentionReference   bool     `json:"attention_reference"`
	RouterTopKReference  bool     `json:"router_topk_reference"`
	ExpertOutputFixture  bool     `json:"expert_output_fixture"`
	CompleteRuntimeTrace bool     `json:"complete_runtime_trace"`
	Missing              []string `json:"missing,omitempty"`
}

func LoadReferenceMetadata(path string) (ReferenceMetadata, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return ReferenceMetadata{}, err
	}
	var meta ReferenceMetadata
	if err := json.Unmarshal(data, &meta); err == nil && meta.Config.ModelType != "" {
		return meta, meta.Validate()
	}
	cfg, err := ParseConfig(data)
	if err != nil {
		return ReferenceMetadata{}, err
	}
	meta = ReferenceMetadata{Name: "config", Config: cfg}
	return meta, nil
}

func (m ReferenceMetadata) Coverage() ReferenceCoverage {
	cov := ReferenceCoverage{ConfigMetadata: m.Config.ModelType == ModelType}
	if cov.ConfigMetadata {
		_, err := NewRuntimePlan(m.Config)
		cov.RuntimePlan = err == nil
	}
	cov.TensorCoverage = m.Tensors.Total > 0
	cov.TensorReadiness = m.Tensors.Readiness.Ready
	cov.CompleteRuntimeTrace = cov.ConfigMetadata && cov.RuntimePlan && cov.TensorReadiness && cov.TokenizationFixture && cov.FirstTokenLogits && cov.ConvLayerReference && cov.AttentionReference && cov.RouterTopKReference && cov.ExpertOutputFixture
	if !cov.ConfigMetadata {
		cov.Missing = append(cov.Missing, "config_metadata")
	}
	if !cov.RuntimePlan {
		cov.Missing = append(cov.Missing, "runtime_plan")
	}
	if !cov.TensorCoverage {
		cov.Missing = append(cov.Missing, "tensor_coverage")
	}
	if !cov.TensorReadiness {
		cov.Missing = append(cov.Missing, "tensor_readiness")
	}
	for _, name := range []string{"tokenization_fixture", "first_token_logits", "conv_layer_reference", "attention_reference", "router_topk_reference", "expert_output_fixture"} {
		cov.Missing = append(cov.Missing, name)
	}
	return cov
}

func (m ReferenceMetadata) Validate() error {
	if err := m.Config.Validate(); err != nil {
		return err
	}
	if m.Tensors.Total < 0 || m.Tensors.Other < 0 {
		return fmt.Errorf("invalid negative tensor coverage: %+v", m.Tensors)
	}
	return nil
}
