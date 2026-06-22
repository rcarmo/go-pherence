package minicpmv

import "testing"

func TestMiniCPMOFixtureMetadata(t *testing.T) {
	meta, err := LoadMetadata("testdata/minicpmo_fixture")
	if err != nil {
		t.Fatalf("LoadMetadata fixture: %v", err)
	}
	if meta.Summary.ModelType != "minicpm-o" || meta.Summary.AudioModelType != "whisper_encoder" || meta.Summary.NumQuery != 64 || meta.Summary.ResamplerGrid != 8 {
		t.Fatalf("bad fixture summary: %+v", meta.Summary)
	}
	if meta.Processor == nil || meta.Processor.NormalizedSize != 448 || meta.Tokenizer == nil || meta.Generation == nil {
		t.Fatalf("missing sidecars: processor=%+v tokenizer=%+v generation=%+v", meta.Processor, meta.Tokenizer, meta.Generation)
	}
	if meta.SpecialTokenIDs == nil || meta.SpecialTokenIDs.ImagePatch != 151642 || meta.AudioSpecialTokenIDs == nil || meta.AudioSpecialTokenIDs.AudioPatch != 151653 {
		t.Fatalf("bad special tokens: image=%+v audio=%+v", meta.SpecialTokenIDs, meta.AudioSpecialTokenIDs)
	}
	if !meta.RuntimePlan.ConfigReady || !meta.RuntimePlan.ProcessorReady || !meta.RuntimePlan.TokenizerReady || !meta.RuntimePlan.SpecialTokensReady || !meta.RuntimePlan.ImagePreprocessReady || !meta.RuntimePlan.PromptPlanningReady {
		t.Fatalf("fixture metadata readiness failed: %+v", meta.RuntimePlan)
	}
	if !meta.ReadinessReport.MetadataReady || meta.ReadinessReport.RuntimeReady {
		t.Fatalf("bad fixture readiness report: %+v", meta.ReadinessReport)
	}
	prompt, err := BuildMultiModalPromptPlan(meta.Summary, meta.Tokenizer, MultiModalPromptOptions{Question: "Describe both.", Images: 1, Audios: 1})
	if err != nil || prompt.Text == "" || prompt.ImagePrompt == nil || prompt.AudioPrompt == nil {
		t.Fatalf("fixture multimodal prompt preview failed: prompt=%+v err=%v", prompt, err)
	}
}
