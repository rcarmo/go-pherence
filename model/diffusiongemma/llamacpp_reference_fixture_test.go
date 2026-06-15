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

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestGGUFHi1x1ParityStatusDocumentsCurrentBlocker(t *testing.T) {
	data, err := os.ReadFile("testdata/gguf_hi_1x1_parity_status.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Status                     string   `json:"status"`
		LlamaCppRawPromptIDs       []int    `json:"llamacpp_raw_prompt_ids"`
		LlamaCppTemplatedPromptIDs []int    `json:"llamacpp_templated_prompt_ids"`
		GoPromptIDs                []int    `json:"go_prompt_ids"`
		GoGeneratedIDs             []int    `json:"go_generated_ids"`
		GoGeneratedTokens          []string `json:"go_generated_tokens"`
		KnownMatches               []string `json:"known_matches"`
		KnownDifferences           []string `json:"known_differences"`
		NextRequiredAction         string   `json:"next_required_action"`
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Status != "partial_token_match_blocked" {
		t.Fatalf("status=%q want partial_token_match_blocked until full sampler/canvas parity is aligned", fixture.Status)
	}
	if !equalInts(fixture.LlamaCppRawPromptIDs, []int{2, 2202}) {
		t.Fatalf("llama.cpp raw prompt IDs=%v want [2 2202]", fixture.LlamaCppRawPromptIDs)
	}
	wantPrompt := []int{2, 105, 2364, 107, 2202, 106, 107, 105, 4368, 107}
	if !equalInts(fixture.LlamaCppTemplatedPromptIDs, wantPrompt) || !equalInts(fixture.GoPromptIDs, wantPrompt) {
		t.Fatalf("templated prompt IDs diverged: llama.cpp=%v Go=%v want=%v", fixture.LlamaCppTemplatedPromptIDs, fixture.GoPromptIDs, wantPrompt)
	}
	if len(fixture.GoGeneratedIDs) != 1 || fixture.GoGeneratedIDs[0] != 100 || len(fixture.GoGeneratedTokens) != 1 || fixture.GoGeneratedTokens[0] != "<|channel>" {
		t.Fatalf("Go generated fixture lost first-token match details: ids=%v tokens=%v", fixture.GoGeneratedIDs, fixture.GoGeneratedTokens)
	}
	if len(fixture.KnownMatches) < 3 || len(fixture.KnownDifferences) < 3 || !strings.Contains(fixture.NextRequiredAction, "full denoised canvas") {
		t.Fatalf("parity blocker is underspecified: %+v", fixture)
	}
}
