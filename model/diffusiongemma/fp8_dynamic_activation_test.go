package diffusiongemma

import "testing"

func TestFP8DynamicActivationIsOptIn(t *testing.T) {
	for _, v := range []string{"", "0", "false", "no", "off"} {
		t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_FP8_DYNAMIC_ACT", v)
		if diffusionGemmaFP8DynamicActivationEnabled() {
			t.Fatalf("dynamic activation enabled for %q", v)
		}
	}
	for _, v := range []string{"1", "true", "yes", "on"} {
		t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_FP8_DYNAMIC_ACT", v)
		if !diffusionGemmaFP8DynamicActivationEnabled() {
			t.Fatalf("dynamic activation disabled for %q", v)
		}
	}
}
