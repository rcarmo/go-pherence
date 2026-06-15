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

func firstIDMismatch(ref, got []int) (index, refID, gotID int) {
	n := len(ref)
	if len(got) < n {
		n = len(got)
	}
	for i := 0; i < n; i++ {
		if ref[i] != got[i] {
			return i, ref[i], got[i]
		}
	}
	if len(ref) != len(got) {
		if len(ref) < len(got) {
			return len(ref), -1, got[len(ref)]
		}
		return len(got), ref[len(got)], -1
	}
	return -1, -1, -1
}

func TestLlamaCppGGUFHi1x1GoldenResponseIDs(t *testing.T) {
	data, err := os.ReadFile("testdata/gguf_hi_1x1_parity_status.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		LlamaCppTemplatedPromptIDs []int          `json:"llamacpp_templated_prompt_ids"`
		LlamaCppResponseIDs        []int          `json:"llamacpp_response_ids"`
		LlamaCppObservedSteps      int            `json:"llamacpp_observed_entropy_bound_steps"`
		LlamaCppStepDiagnostics    []CanvasStep   `json:"llamacpp_step_diagnostics"`
		LlamaCppTopProbes          []EntropyProbe `json:"llamacpp_first_step_top_probes"`
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
	if fixture.LlamaCppObservedSteps != 11 || len(fixture.LlamaCppStepDiagnostics) != 11 {
		t.Fatalf("llama.cpp observed entropy-bound steps=%d diagnostics=%d want 11", fixture.LlamaCppObservedSteps, len(fixture.LlamaCppStepDiagnostics))
	}
	first := fixture.LlamaCppStepDiagnostics[0]
	last := fixture.LlamaCppStepDiagnostics[len(fixture.LlamaCppStepDiagnostics)-1]
	if first.Step != 48 || first.Accepted != 20 || first.MeanEntropy < 1.30 || first.MeanEntropy > 1.31 || first.FirstArgmax != 100 || first.FirstSampled != 100 || !first.FirstAccepted || first.FirstEntropy < 0.007 || first.FirstEntropy > 0.008 || first.MaxEntropy < 4.03 || first.MaxEntropy > 4.04 || first.MaxEntropyPos != 28 || first.Stopped {
		t.Fatalf("bad llama.cpp first step diagnostics: %+v", first)
	}
	if last.Step != 38 || last.Accepted != 254 || last.FirstArgmax != 100 || last.FirstSampled != 100 || !last.FirstAccepted || last.Held != 1 || !last.Confident || !last.Stopped {
		t.Fatalf("bad llama.cpp final step diagnostics: %+v", last)
	}
	if len(fixture.LlamaCppTopProbes) != 2 {
		t.Fatalf("llama.cpp top probes=%d want 2", len(fixture.LlamaCppTopProbes))
	}
	if p := fixture.LlamaCppTopProbes[0]; p.Position != 9 || p.Argmax != 107 || p.Sampled != 107 || p.Accepted || p.Entropy < 1.16 || p.Entropy > 1.18 || !equalInts(p.TopIDs, []int{107, 3056, 1174, 108, 140}) {
		t.Fatalf("bad llama.cpp pos9 top probe: %+v", p)
	}
	if p := fixture.LlamaCppTopProbes[1]; p.Position != 28 || p.Argmax != 139 || p.Sampled != 139 || p.Accepted || p.Entropy < 4.03 || p.Entropy > 4.04 || !equalInts(p.TopIDs, []int{139, 236829, 236761, 107, 140}) {
		t.Fatalf("bad llama.cpp pos28 top probe: %+v", p)
	}
}

func TestGGUFHi1x1GoTrimmedOutputComparisonGate(t *testing.T) {
	data, err := os.ReadFile("testdata/gguf_hi_1x1_parity_status.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		LlamaCppTemplatedPromptIDs []int        `json:"llamacpp_templated_prompt_ids"`
		LlamaCppResponseIDs        []int        `json:"llamacpp_response_ids"`
		LlamaCppStepDiagnostics    []CanvasStep `json:"llamacpp_step_diagnostics"`
		GoPromptIDs                []int        `json:"go_prompt_ids"`
		GoGeneratedIDs             []int        `json:"go_generated_ids"`
		GoTrimCut                  int          `json:"go_trim_cut"`
		GoObservedSteps            int          `json:"go_observed_steps"`
		GoStepDiagnostics          []CanvasStep `json:"go_step_diagnostics"`
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if !equalInts(fixture.GoPromptIDs, fixture.LlamaCppTemplatedPromptIDs) {
		t.Fatalf("prompt IDs differ: go=%v llama=%v", fixture.GoPromptIDs, fixture.LlamaCppTemplatedPromptIDs)
	}
	if len(fixture.GoGeneratedIDs) != fixture.GoTrimCut {
		t.Fatalf("go generated len=%d trim_cut=%d", len(fixture.GoGeneratedIDs), fixture.GoTrimCut)
	}
	if equalInts(fixture.GoGeneratedIDs, fixture.LlamaCppResponseIDs) {
		t.Fatalf("fixture unexpectedly matches; promote this test to an equality golden and update parity status")
	}
	idx, ref, got := firstIDMismatch(fixture.LlamaCppResponseIDs, fixture.GoGeneratedIDs)
	if idx != 3 || ref != 818 || got != 101 {
		t.Fatalf("first mismatch index/ref/go=%d/%d/%d want 3/818/101", idx, ref, got)
	}
	if len(fixture.LlamaCppResponseIDs) != 64 || len(fixture.GoGeneratedIDs) != 13 || fixture.GoObservedSteps != 2 {
		t.Fatalf("unexpected comparison dimensions: ref=%d go=%d steps=%d", len(fixture.LlamaCppResponseIDs), len(fixture.GoGeneratedIDs), fixture.GoObservedSteps)
	}
	if len(fixture.LlamaCppStepDiagnostics) == 0 || len(fixture.GoStepDiagnostics) == 0 {
		t.Fatalf("missing step diagnostics: llama=%d go=%d", len(fixture.LlamaCppStepDiagnostics), len(fixture.GoStepDiagnostics))
	}
	lf, gf := fixture.LlamaCppStepDiagnostics[0], fixture.GoStepDiagnostics[0]
	if lf.Step != gf.Step || lf.Accepted != 20 || gf.Accepted != 243 || lf.MeanEntropy < 1.30 || gf.MeanEntropy > 0.016 || lf.FirstArgmax != 100 || gf.FirstArgmax != 100 || lf.FirstSampled != 100 || gf.FirstSampled != 100 || !lf.FirstAccepted || gf.FirstAccepted || lf.MaxEntropyPos != 28 || gf.MaxEntropyPos != 9 || lf.MaxEntropy < 4.03 || gf.MaxEntropy > 1.03 {
		t.Fatalf("unexpected first-step gap: llama=%+v go=%+v", lf, gf)
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
		Row28LayerTraceSummary     struct {
			Layer0 struct {
				LlamaRMS float64 `json:"llama_rms"`
				GoRMS    float64 `json:"go_rms"`
			} `json:"layer0_l_out"`
			Layer29 struct {
				LlamaRMS    float64 `json:"llama_rms"`
				GoRMS       float64 `json:"go_rms"`
				LlamaMaxAbs float64 `json:"llama_max_abs"`
				GoMaxAbs    float64 `json:"go_max_abs"`
			} `json:"layer29_l_out"`
		} `json:"row28_layer_trace_summary"`
		KnownMatches       []string `json:"known_matches"`
		KnownDifferences   []string `json:"known_differences"`
		NextRequiredAction string   `json:"next_required_action"`
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
	if fixture.Row28LayerTraceSummary.Layer0.LlamaRMS < 1.79 || fixture.Row28LayerTraceSummary.Layer0.GoRMS < 1.79 || fixture.Row28LayerTraceSummary.Layer29.LlamaRMS > 0.64 || fixture.Row28LayerTraceSummary.Layer29.GoRMS < 1.12 || fixture.Row28LayerTraceSummary.Layer29.GoMaxAbs < 39 || fixture.Row28LayerTraceSummary.Layer29.LlamaMaxAbs > 19 {
		t.Fatalf("row28 layer trace summary lost divergence target: %+v", fixture.Row28LayerTraceSummary)
	}
	if len(fixture.KnownMatches) < 4 || len(fixture.KnownDifferences) < 5 || !strings.Contains(fixture.NextRequiredAction, "later layers") {
		t.Fatalf("parity blocker is underspecified: %+v", fixture)
	}
}
