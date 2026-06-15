package model

import "fmt"

// MTPGraphCapabilities reports the current Gemma4 MTP backend-graph support
// surface. It deliberately separates production-safe graph pieces from gated
// experimental full-layer batch lowering.
type MTPGraphCapabilities struct {
	DrafterLoader                  bool `json:"drafter_loader"`
	HiddenStateDrafterLoop         bool `json:"hidden_state_drafter_loop"`
	VerifierPlan                   bool `json:"verifier_plan"`
	VerifierBatchInputs            bool `json:"verifier_batch_inputs"`
	VerifierAttentionPlan          bool `json:"verifier_attention_plan"`
	VerifierScratchPlan            bool `json:"verifier_scratch_plan"`
	VerifierTailBatch              bool `json:"verifier_tail_batch"`
	VerifierSequentialLayers       bool `json:"verifier_sequential_layers"`
	VerifierBatchQKVProjection     bool `json:"verifier_batch_qkv_projection"`
	VerifierBatchQATProjection     bool `json:"verifier_batch_qat_projection"`
	VerifierBatchPLI               bool `json:"verifier_batch_pli"`
	VerifierBatchLayers            bool `json:"verifier_batch_layers"`
	VerifierBatchLayersGated       bool `json:"verifier_batch_layers_gated"`
	VerifierCompressedKVStaging    bool `json:"verifier_compressed_kv_staging"`
	AcceptanceBonusSemantics       bool `json:"acceptance_bonus_semantics"`
	GraphKVCommit                  bool `json:"graph_kv_commit"`
	AdaptiveDraftPolicy            bool `json:"adaptive_draft_policy"`
	PromptContextAPI               bool `json:"prompt_context_api"`
	ExternalKVRefresh              bool `json:"external_kv_refresh"`
	ExactTokenBudget               bool `json:"exact_token_budget"`
	ExperimentalGenerationWiring   bool `json:"experimental_generation_wiring"`
	ReadyForExperimentalGeneration bool `json:"ready_for_experimental_generation"`
	PublicGenerationWiring         bool `json:"public_generation_wiring"`
	ReadyForPublicGeneration       bool `json:"ready_for_public_generation"`
}

func Gemma4MTPGraphCapabilities() MTPGraphCapabilities {
	caps := MTPGraphCapabilities{
		DrafterLoader:                true,
		HiddenStateDrafterLoop:       true,
		VerifierPlan:                 true,
		VerifierBatchInputs:          true,
		VerifierAttentionPlan:        true,
		VerifierScratchPlan:          true,
		VerifierTailBatch:            true,
		VerifierSequentialLayers:     true,
		VerifierBatchQKVProjection:   true,
		VerifierBatchQATProjection:   true,
		VerifierBatchPLI:             true,
		VerifierBatchLayers:          mtpVerifierBatchLayerLoweringEnabled(),
		VerifierBatchLayersGated:     true,
		VerifierCompressedKVStaging:  false,
		AcceptanceBonusSemantics:     true,
		GraphKVCommit:                true,
		AdaptiveDraftPolicy:          true,
		PromptContextAPI:             true,
		ExternalKVRefresh:            true,
		ExactTokenBudget:             true,
		ExperimentalGenerationWiring: true,
		PublicGenerationWiring:       false,
	}
	caps.ReadyForExperimentalGeneration = caps.ExperimentalGenerationWiring && caps.PromptContextAPI && caps.ExternalKVRefresh && caps.ExactTokenBudget && caps.GraphKVCommit && caps.AcceptanceBonusSemantics && caps.HiddenStateDrafterLoop && caps.AdaptiveDraftPolicy
	caps.ReadyForPublicGeneration = caps.PublicGenerationWiring && caps.ReadyForExperimentalGeneration && caps.VerifierCompressedKVStaging && (!caps.VerifierBatchLayersGated || caps.VerifierBatchLayers)
	return caps
}

func (c MTPGraphCapabilities) Validate() error {
	wantExperimental := c.ExperimentalGenerationWiring && c.PromptContextAPI && c.ExternalKVRefresh && c.ExactTokenBudget && c.GraphKVCommit && c.AcceptanceBonusSemantics && c.HiddenStateDrafterLoop && c.AdaptiveDraftPolicy
	if c.ReadyForExperimentalGeneration != wantExperimental {
		return fmt.Errorf("MTP experimental readiness=%v, want %v from capability flags", c.ReadyForExperimentalGeneration, wantExperimental)
	}
	wantPublic := c.PublicGenerationWiring && c.ReadyForExperimentalGeneration && c.VerifierCompressedKVStaging && (!c.VerifierBatchLayersGated || c.VerifierBatchLayers)
	if c.ReadyForPublicGeneration != wantPublic {
		return fmt.Errorf("MTP public readiness=%v, want %v from public wiring and experimental readiness", c.ReadyForPublicGeneration, wantPublic)
	}
	return nil
}

func (c MTPGraphCapabilities) MissingForPublicGeneration() []string {
	var missing []string
	if !c.PublicGenerationWiring {
		missing = append(missing, "public_generation_wiring")
	}
	if !c.VerifierCompressedKVStaging {
		missing = append(missing, "compressed_kv_verifier_staging")
	}
	if !c.ReadyForPublicGeneration && c.VerifierBatchLayersGated && !c.VerifierBatchLayers {
		missing = append(missing, "full_layer_batch_verifier_default_enablement")
	}
	return missing
}
