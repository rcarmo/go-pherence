package minicpmv

import (
	"encoding/json"
	"os"
	"testing"
)

type fixtureExpectedSummary struct {
	ModelType         string `json:"model_type"`
	Architecture      string `json:"architecture"`
	TextModelType     string `json:"text_model_type"`
	VisionModelType   string `json:"vision_model_type"`
	AudioModelType    string `json:"audio_model_type"`
	HiddenSize        int    `json:"hidden_size"`
	VisionHiddenSize  int    `json:"vision_hidden_size"`
	AudioHiddenSize   int    `json:"audio_hidden_size"`
	NumQuery          int    `json:"num_query"`
	ResamplerGrid     int    `json:"resampler_grid"`
	ImagePatchTokenID int    `json:"image_patch_token_id"`
	AudioPatchTokenID int    `json:"audio_patch_token_id"`
	MetadataReady     bool   `json:"metadata_ready"`
	RuntimeReady      bool   `json:"runtime_ready"`
}

func TestMiniCPMOFixtureMetadata(t *testing.T) {
	meta, err := LoadMetadata("testdata/minicpmo_fixture")
	if err != nil {
		t.Fatalf("LoadMetadata fixture: %v", err)
	}
	expected := readFixtureExpectedSummary(t)
	if meta.Summary.ModelType != expected.ModelType || meta.Summary.Architecture != expected.Architecture || meta.Summary.TextModelType != expected.TextModelType || meta.Summary.VisionModelType != expected.VisionModelType || meta.Summary.AudioModelType != expected.AudioModelType || meta.Summary.HiddenSize != expected.HiddenSize || meta.Summary.VisionHiddenSize != expected.VisionHiddenSize || meta.Summary.AudioHiddenSize != expected.AudioHiddenSize || meta.Summary.NumQuery != expected.NumQuery || meta.Summary.ResamplerGrid != expected.ResamplerGrid {
		t.Fatalf("bad fixture summary: got=%+v expected=%+v", meta.Summary, expected)
	}
	if meta.Processor == nil || meta.Processor.NormalizedSize != 448 || meta.Tokenizer == nil || meta.Generation == nil {
		t.Fatalf("missing sidecars: processor=%+v tokenizer=%+v generation=%+v", meta.Processor, meta.Tokenizer, meta.Generation)
	}
	if meta.SpecialTokenIDs == nil || meta.SpecialTokenIDs.ImagePatch != expected.ImagePatchTokenID || meta.AudioSpecialTokenIDs == nil || meta.AudioSpecialTokenIDs.AudioPatch != expected.AudioPatchTokenID {
		t.Fatalf("bad special tokens: image=%+v audio=%+v", meta.SpecialTokenIDs, meta.AudioSpecialTokenIDs)
	}
	if !meta.RuntimePlan.ConfigReady || !meta.RuntimePlan.ProcessorReady || !meta.RuntimePlan.TokenizerReady || !meta.RuntimePlan.SpecialTokensReady || !meta.RuntimePlan.ImagePreprocessReady || !meta.RuntimePlan.PromptPlanningReady {
		t.Fatalf("fixture metadata readiness failed: %+v", meta.RuntimePlan)
	}
	if meta.ReadinessReport.MetadataReady != expected.MetadataReady || meta.ReadinessReport.RuntimeReady != expected.RuntimeReady {
		t.Fatalf("bad fixture readiness report: got=%+v expected=%+v", meta.ReadinessReport, expected)
	}
	prompt, err := BuildMultiModalPromptPlan(meta.Summary, meta.Tokenizer, MultiModalPromptOptions{Question: "Describe both.", Images: 1, Audios: 1})
	if err != nil || prompt.Text == "" || prompt.ImagePrompt == nil || prompt.AudioPrompt == nil {
		t.Fatalf("fixture multimodal prompt preview failed: prompt=%+v err=%v", prompt, err)
	}
}

func readFixtureExpectedSummary(t *testing.T) fixtureExpectedSummary {
	t.Helper()
	b, err := os.ReadFile("testdata/minicpmo_fixture/expected_summary.json")
	if err != nil {
		t.Fatal(err)
	}
	var out fixtureExpectedSummary
	if err := json.Unmarshal(b, &out); err != nil {
		t.Fatal(err)
	}
	return out
}
