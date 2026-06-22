package minicpmv

import "testing"

func TestMiniCPMOFixtureMetadata(t *testing.T) {
	meta, err := LoadMiniCPMOFixtureMetadata()
	if err != nil {
		t.Fatalf("LoadMetadata fixture: %v", err)
	}
	expected, err := LoadMiniCPMOFixtureExpectedSummary()
	if err != nil {
		t.Fatalf("LoadMiniCPMOFixtureExpectedSummary: %v", err)
	}
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
