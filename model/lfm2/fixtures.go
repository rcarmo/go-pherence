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
	Name           string                 `json:"name,omitempty"`
	Config         Config                 `json:"config"`
	Tensors        TensorCoverage         `json:"tensors,omitempty"`
	References     *ReferenceSummaries    `json:"references,omitempty"`
	RuntimeRequest *RuntimeRequestSummary `json:"runtime_request,omitempty"`
}

type RuntimeRequestSummary struct {
	PromptTokens   int   `json:"prompt_tokens"`
	MaxNewTokens   int   `json:"max_new_tokens"`
	MaxSequence    int   `json:"max_sequence"`
	BytesPerFloat  int   `json:"bytes_per_float"`
	KVBytes        int64 `json:"kv_bytes"`
	ConvStateBytes int64 `json:"conv_state_bytes"`
}

type ReferenceSummaries struct {
	Tokenization   *TokenizationReference `json:"tokenization,omitempty"`
	FirstToken     *FirstTokenReference   `json:"first_token,omitempty"`
	ConvLayer      *LayerReference        `json:"conv_layer,omitempty"`
	AttentionLayer *LayerReference        `json:"attention_layer,omitempty"`
	RouterTopK     *RouterTopKReference   `json:"router_topk,omitempty"`
	ExpertOutput   *LayerReference        `json:"expert_output,omitempty"`
}

type TokenizationReference struct {
	Text   string   `json:"text"`
	Tokens []uint32 `json:"tokens"`
}

type FirstTokenReference struct {
	TokenID       uint32 `json:"token_id"`
	LogitChecksum string `json:"logit_checksum,omitempty"`
}

type LayerReference struct {
	Layer      int     `json:"layer"`
	Checksum   string  `json:"checksum,omitempty"`
	MaxAbsDiff float64 `json:"max_abs_diff,omitempty"`
}

type RouterTopKReference struct {
	Layer      int    `json:"layer"`
	ExpertIDs  []int  `json:"expert_ids"`
	WeightHash string `json:"weight_hash,omitempty"`
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
	RuntimeRequest       bool     `json:"runtime_request"`
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
		if meta.Config.HeadDim == 0 && meta.Config.NumAttentionHeads > 0 {
			meta.Config.HeadDim = meta.Config.HiddenSize / meta.Config.NumAttentionHeads
		}
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
	if m.References != nil {
		cov.TokenizationFixture = m.References.Tokenization != nil && len(m.References.Tokenization.Tokens) > 0
		cov.FirstTokenLogits = m.References.FirstToken != nil
		cov.ConvLayerReference = m.References.ConvLayer != nil
		cov.AttentionReference = m.References.AttentionLayer != nil
		cov.RouterTopKReference = m.References.RouterTopK != nil && len(m.References.RouterTopK.ExpertIDs) > 0
		cov.ExpertOutputFixture = m.References.ExpertOutput != nil
	}
	cov.RuntimeRequest = m.RuntimeRequest != nil
	cov.CompleteRuntimeTrace = cov.ConfigMetadata && cov.RuntimePlan && cov.TensorReadiness && cov.TokenizationFixture && cov.FirstTokenLogits && cov.ConvLayerReference && cov.AttentionReference && cov.RouterTopKReference && cov.ExpertOutputFixture && cov.RuntimeRequest
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
	if !cov.TokenizationFixture {
		cov.Missing = append(cov.Missing, "tokenization_fixture")
	}
	if !cov.FirstTokenLogits {
		cov.Missing = append(cov.Missing, "first_token_logits")
	}
	if !cov.ConvLayerReference {
		cov.Missing = append(cov.Missing, "conv_layer_reference")
	}
	if !cov.AttentionReference {
		cov.Missing = append(cov.Missing, "attention_reference")
	}
	if !cov.RouterTopKReference {
		cov.Missing = append(cov.Missing, "router_topk_reference")
	}
	if !cov.ExpertOutputFixture {
		cov.Missing = append(cov.Missing, "expert_output_fixture")
	}
	if !cov.RuntimeRequest {
		cov.Missing = append(cov.Missing, "runtime_request")
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
	if m.RuntimeRequest != nil {
		if m.RuntimeRequest.PromptTokens <= 0 || m.RuntimeRequest.MaxNewTokens <= 0 || m.RuntimeRequest.MaxSequence != m.RuntimeRequest.PromptTokens+m.RuntimeRequest.MaxNewTokens || m.RuntimeRequest.BytesPerFloat <= 0 {
			return fmt.Errorf("invalid LFM2 runtime request summary: %+v", m.RuntimeRequest)
		}
		if m.RuntimeRequest.KVBytes < 0 || m.RuntimeRequest.ConvStateBytes <= 0 {
			return fmt.Errorf("invalid LFM2 runtime request sizing summary: %+v", m.RuntimeRequest)
		}
	}
	if m.References != nil {
		if r := m.References.Tokenization; r != nil && (r.Text == "" || len(r.Tokens) == 0) {
			return fmt.Errorf("invalid LFM2 tokenization reference: %+v", r)
		}
		if r := m.References.ConvLayer; r != nil && r.Layer < 0 {
			return fmt.Errorf("invalid LFM2 conv layer reference: %+v", r)
		}
		if r := m.References.AttentionLayer; r != nil && r.Layer < 0 {
			return fmt.Errorf("invalid LFM2 attention layer reference: %+v", r)
		}
		if r := m.References.ExpertOutput; r != nil && r.Layer < 0 {
			return fmt.Errorf("invalid LFM2 expert output reference: %+v", r)
		}
		if r := m.References.RouterTopK; r != nil {
			if r.Layer < 0 || len(r.ExpertIDs) == 0 {
				return fmt.Errorf("invalid LFM2 router top-k reference: %+v", r)
			}
			for _, id := range r.ExpertIDs {
				if id < 0 || id >= m.Config.NumExperts {
					return fmt.Errorf("invalid LFM2 router expert id=%d experts=%d", id, m.Config.NumExperts)
				}
			}
		}
	}
	return nil
}
