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

func TestLlamaCppGGUFHi1x1GoldenResponseIDs(t *testing.T) {
	data, err := os.ReadFile("testdata/gguf_hi_1x1_parity_status.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		LlamaCppTemplatedPromptIDs []int `json:"llamacpp_templated_prompt_ids"`
		LlamaCppResponseIDs        []int `json:"llamacpp_response_ids"`
		LlamaCppObservedSteps      int   `json:"llamacpp_observed_entropy_bound_steps"`
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	wantPrompt := []int{2, 105, 2364, 107, 2202, 106, 107, 105, 4368, 107}
	wantResponse := []int{100, 45518, 107, 818, 2430, 1176, 623, 2202, 3056, 107, 2094, 563, 496, 26227, 236761, 107, 236777, 1374, 8932, 124954, 532, 2729, 10686, 236761, 108, 21817, 236787, 107, 236770, 236761, 22168, 1063, 6791, 9259, 9332, 653, 623, 10979, 993, 236888, 4248, 107, 236778, 236761, 29020, 623, 3910, 740, 564, 1601, 611, 3124, 126584, 101, 9259, 236888, 2088, 740, 564, 1601, 611, 3124, 236881, 107}
	if !equalInts(fixture.LlamaCppTemplatedPromptIDs, wantPrompt) {
		t.Fatalf("llama.cpp prompt IDs=%v want %v", fixture.LlamaCppTemplatedPromptIDs, wantPrompt)
	}
	if !equalInts(fixture.LlamaCppResponseIDs, wantResponse) {
		t.Fatalf("llama.cpp response IDs changed:\n got=%v\nwant=%v", fixture.LlamaCppResponseIDs, wantResponse)
	}
	if fixture.LlamaCppObservedSteps != 11 {
		t.Fatalf("llama.cpp observed entropy-bound steps=%d want 11", fixture.LlamaCppObservedSteps)
	}
}

func TestGGUFHi1x1ParityStatusDocumentsCurrentBlocker(t *testing.T) {
	data, err := os.ReadFile("testdata/gguf_hi_1x1_parity_status.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Status                     string       `json:"status"`
		LlamaCppRawPromptIDs       []int        `json:"llamacpp_raw_prompt_ids"`
		LlamaCppTemplatedPromptIDs []int        `json:"llamacpp_templated_prompt_ids"`
		GoPromptIDs                []int        `json:"go_prompt_ids"`
		GoGeneratedIDs             []int        `json:"go_generated_ids"`
		GoGeneratedTokens          []string     `json:"go_generated_tokens"`
		GoTrimCut                  int          `json:"go_trim_cut"`
		GoTrimReason               string       `json:"go_trim_reason"`
		GoObservedSteps            int          `json:"go_observed_steps"`
		GoStepDiagnostics          []CanvasStep `json:"go_step_diagnostics"`
		KnownMatches               []string     `json:"known_matches"`
		KnownDifferences           []string     `json:"known_differences"`
		NextRequiredAction         string       `json:"next_required_action"`
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Status != "trimmed_output_mismatch_blocked" {
		t.Fatalf("status=%q want trimmed_output_mismatch_blocked until full sampler/canvas parity is aligned", fixture.Status)
	}
	if !equalInts(fixture.LlamaCppRawPromptIDs, []int{2, 2202}) {
		t.Fatalf("llama.cpp raw prompt IDs=%v want [2 2202]", fixture.LlamaCppRawPromptIDs)
	}
	wantPrompt := []int{2, 105, 2364, 107, 2202, 106, 107, 105, 4368, 107}
	if !equalInts(fixture.LlamaCppTemplatedPromptIDs, wantPrompt) || !equalInts(fixture.GoPromptIDs, wantPrompt) {
		t.Fatalf("templated prompt IDs diverged: llama.cpp=%v Go=%v want=%v", fixture.LlamaCppTemplatedPromptIDs, fixture.GoPromptIDs, wantPrompt)
	}
	wantGenerated := []int{100, 45518, 107, 101, 9259, 236888, 2088, 740, 564, 1601, 611, 3124, 236881}
	if !equalInts(fixture.GoGeneratedIDs, wantGenerated) || len(fixture.GoGeneratedTokens) != len(wantGenerated) || fixture.GoGeneratedTokens[0] != "<|channel>" {
		t.Fatalf("Go generated fixture lost trimmed-output mismatch details: ids=%v tokens=%v", fixture.GoGeneratedIDs, fixture.GoGeneratedTokens)
	}
	if fixture.GoTrimCut != len(wantGenerated) || fixture.GoTrimReason != "eog" || fixture.GoObservedSteps != 2 || len(fixture.GoStepDiagnostics) != 2 || !fixture.GoStepDiagnostics[1].Stopped || fixture.GoStepDiagnostics[1].Held != 1 || !fixture.GoStepDiagnostics[1].Confident {
		t.Fatalf("Go trim/step diagnostics lost: cut=%d reason=%q observed=%d steps=%+v", fixture.GoTrimCut, fixture.GoTrimReason, fixture.GoObservedSteps, fixture.GoStepDiagnostics)
	}
	if len(fixture.KnownMatches) < 4 || len(fixture.KnownDifferences) < 4 || !strings.Contains(fixture.NextRequiredAction, "first-step logits/entropy") {
		t.Fatalf("parity blocker is underspecified: %+v", fixture)
	}
}
