package model

import "testing"

func TestGemma4MTPGraphCapabilitiesDefault(t *testing.T) {
	t.Setenv("GO_PHERENCE_MTP_VERIFIER_BATCH_LAYERS", "")
	caps := Gemma4MTPGraphCapabilities()
	if !caps.DrafterLoader || !caps.HiddenStateDrafterLoop || !caps.VerifierBatchInputs || !caps.VerifierAttentionPlan || !caps.VerifierTailBatch || !caps.VerifierBatchQATProjection || !caps.VerifierBatchPLI || !caps.GraphKVCommit || !caps.AdaptiveDraftPolicy || !caps.PromptContextAPI {
		t.Fatalf("missing implemented capability: %+v", caps)
	}
	if caps.VerifierBatchLayers || !caps.VerifierBatchLayersGated {
		t.Fatalf("batch layer gate state=%+v", caps)
	}
	if caps.PublicGenerationWiring || caps.ReadyForPublicGeneration {
		t.Fatalf("public generation unexpectedly ready: %+v", caps)
	}
	missing := caps.MissingForPublicGeneration()
	if !sameStringSet(missing, []string{"public_generation_wiring", "full_layer_batch_verifier_default_enablement"}) {
		t.Fatalf("missing=%v", missing)
	}
}

func TestGemma4MTPGraphCapabilitiesBatchLayerGate(t *testing.T) {
	t.Setenv("GO_PHERENCE_MTP_VERIFIER_BATCH_LAYERS", "1")
	caps := Gemma4MTPGraphCapabilities()
	if !caps.VerifierBatchLayers || !caps.VerifierBatchLayersGated {
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
