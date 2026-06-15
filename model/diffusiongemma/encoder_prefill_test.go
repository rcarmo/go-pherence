package diffusiongemma

import "testing"

func TestGGUFCPUPrefillOnlyEnabled(t *testing.T) {
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_GGUF_CPU_PREFILL", "")
	if diffusionGemmaGGUFCPUPrefillOnlyEnabled() {
		t.Fatal("GGUF CPU prefill-only should default off")
	}
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_GGUF_CPU_PREFILL", "1")
	if !diffusionGemmaGGUFCPUPrefillOnlyEnabled() {
		t.Fatal("GGUF CPU prefill-only opt-in not honored")
	}
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_GGUF_CPU_PREFILL", "false")
	if diffusionGemmaGGUFCPUPrefillOnlyEnabled() {
		t.Fatal("GGUF CPU prefill-only false should disable")
	}
}
