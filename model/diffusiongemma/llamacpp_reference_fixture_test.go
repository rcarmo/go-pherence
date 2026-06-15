package diffusiongemma

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
)

func TestLlamaCppGGUFHi1x1ReferenceFixture(t *testing.T) {
	data, err := os.ReadFile("testdata/llamacpp_gguf_hi_1x1_reference.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Reference                 string `json:"reference"`
		Prompt                    string `json:"prompt"`
		NPredict                  int    `json:"n_predict"`
		Seed                      int    `json:"seed"`
		CanvasLength              int    `json:"canvas_length"`
		ObservedEntropyBoundSteps int    `json:"observed_entropy_bound_steps"`
		DecodedResponse           string `json:"decoded_response"`
		HFReferenceEnv            struct {
			DiffusionGemmaClassPresent bool   `json:"diffusiongemma_class_present"`
			LocalFP8LoadBlocker        string `json:"local_fp8_load_blocker"`
		} `json:"hf_reference_env"`
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Reference != "llama.cpp llama-diffusion-cli" || fixture.Prompt != "hi" || fixture.NPredict != 1 || fixture.Seed != 1 || fixture.CanvasLength != 256 {
		t.Fatalf("unexpected fixture header: %+v", fixture)
	}
	if fixture.ObservedEntropyBoundSteps != 11 {
		t.Fatalf("observed entropy-bound steps=%d want 11", fixture.ObservedEntropyBoundSteps)
	}
	if !strings.Contains(fixture.DecodedResponse, "Hello! How can I help you today?") {
		t.Fatalf("decoded response missing greeting: %q", fixture.DecodedResponse)
	}
	if !fixture.HFReferenceEnv.DiffusionGemmaClassPresent || fixture.HFReferenceEnv.LocalFP8LoadBlocker == "" {
		t.Fatalf("HF reference environment/blocker not captured: %+v", fixture.HFReferenceEnv)
	}
}
