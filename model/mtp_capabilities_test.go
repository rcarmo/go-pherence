package model

import "testing"

func TestGemma4MTPGraphCapabilitiesDefault(t *testing.T) {
	t.Setenv("GO_PHERENCE_MTP_VERIFIER_BATCH_LAYERS", "")
	caps := Gemma4MTPGraphCapabilities()
	if !caps.DrafterLoader || !caps.HiddenStateDrafterLoop || !caps.VerifierBatchInputs || !caps.VerifierAttentionPlan || !caps.VerifierTailBatch || !caps.VerifierBatchQATProjection || !caps.VerifierBatchPLI || !caps.GraphKVCommit || !caps.AdaptiveDraftPolicy || !caps.PromptContextAPI || !caps.ExternalKVRefresh || !caps.ExactTokenBudget || !caps.ExperimentalGenerationWiring {
		t.Fatalf("missing implemented capability: %+v", caps)
	}
	if !caps.VerifierCompressedKVStaging {
		t.Fatalf("compressed KV verifier staging unexpectedly not ready: %+v", caps)
	}
	if caps.VerifierBatchLayers || !caps.VerifierBatchLayersGated {
		t.Fatalf("batch layer gate state=%+v", caps)
	}
	if !caps.ReadyForExperimentalGeneration {
		t.Fatalf("experimental generation unexpectedly not ready: %+v", caps)
	}
	if caps.PublicGenerationWiring || caps.ReadyForPublicGeneration {
		t.Fatalf("public generation unexpectedly ready: %+v", caps)
	}
	missing := caps.MissingForPublicGeneration()
	if !sameStringSet(missing, []string{"public_generation_wiring", "full_layer_batch_verifier_default_enablement"}) {
		t.Fatalf("missing=%v", missing)
	}
	if err := caps.Validate(); err != nil {
		t.Fatalf("Validate default caps: %v", err)
	}
	bad := caps
	bad.ReadyForExperimentalGeneration = false
	bad.ExperimentalGenerationWiring = true
	bad.PromptContextAPI = true
	bad.ExternalKVRefresh = true
	bad.ExactTokenBudget = true
	bad.GraphKVCommit = true
	bad.AcceptanceBonusSemantics = true
	bad.HiddenStateDrafterLoop = true
	bad.AdaptiveDraftPolicy = true
	if err := bad.Validate(); err == nil {
		t.Fatal("accepted stale experimental readiness")
	}
	bad = caps
	bad.ReadyForPublicGeneration = true
	if err := bad.Validate(); err == nil {
		t.Fatal("accepted stale public readiness")
	}
	publicBlocked := caps
	publicBlocked.PublicGenerationWiring = true
	publicBlocked.ReadyForPublicGeneration = false
	if err := publicBlocked.Validate(); err != nil {
		t.Fatalf("rejected public wiring blocked only by gated batch layers: %v", err)
	}
	if !sameStringSet(publicBlocked.MissingForPublicGeneration(), []string{"full_layer_batch_verifier_default_enablement"}) {
		t.Fatalf("public blocked missing=%v", publicBlocked.MissingForPublicGeneration())
	}
	bad = publicBlocked
	bad.ReadyForPublicGeneration = true
	if err := bad.Validate(); err == nil {
		t.Fatal("accepted public readiness while batch verifier layers are still gated off")
	}
}

func TestGemma4MTPGraphCapabilitiesBatchLayerGate(t *testing.T) {
	t.Setenv("GO_PHERENCE_MTP_VERIFIER_BATCH_LAYERS", "1")
	caps := Gemma4MTPGraphCapabilities()
	if !caps.VerifierBatchLayers || !caps.VerifierBatchLayersGated || !caps.ReadyForExperimentalGeneration {
		t.Fatalf("batch layer gate state=%+v", caps)
	}
	missing := caps.MissingForPublicGeneration()
	if !sameStringSet(missing, []string{"public_generation_wiring"}) {
		t.Fatalf("missing=%v", missing)
	}
}

func sameStringSet(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	seen := map[string]int{}
	for _, x := range a {
		seen[x]++
	}
	for _, x := range b {
		if seen[x] == 0 {
			return false
		}
		seen[x]--
	}
	return true
}
