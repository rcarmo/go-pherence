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

func TestGGUFHi1x1ParityStatusDocumentsCurrentBlocker(t *testing.T) {
	data, err := os.ReadFile("testdata/gguf_hi_1x1_parity_status.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Status                          string   `json:"status"`
		GoPromptIDs                     []int    `json:"go_prompt_ids"`
		GoGeneratedIDs                  []int    `json:"go_generated_ids"`
		GoGeneratedTokens               []string `json:"go_generated_tokens"`
		LlamaCppDecodedResponseContains string   `json:"llamacpp_decoded_response_contains"`
		KnownDifferences                []string `json:"known_differences"`
		NextRequiredAction              string   `json:"next_required_action"`
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Status != "mismatch_blocked" {
		t.Fatalf("status=%q want mismatch_blocked until prompt/sampler parity is aligned", fixture.Status)
	}
	if len(fixture.GoPromptIDs) != 1 || fixture.GoPromptIDs[0] != 2202 {
		t.Fatalf("Go prompt IDs=%v want [2202]", fixture.GoPromptIDs)
	}
	if len(fixture.GoGeneratedIDs) != 1 || fixture.GoGeneratedIDs[0] != 236761 || len(fixture.GoGeneratedTokens) != 1 || fixture.GoGeneratedTokens[0] != "." {
		t.Fatalf("Go generated fixture lost mismatch details: ids=%v tokens=%v", fixture.GoGeneratedIDs, fixture.GoGeneratedTokens)
	}
	if !strings.Contains(fixture.LlamaCppDecodedResponseContains, "Hello!") || len(fixture.KnownDifferences) < 3 || !strings.Contains(fixture.NextRequiredAction, "prompt IDs") {
		t.Fatalf("parity blocker is underspecified: %+v", fixture)
	}
}
