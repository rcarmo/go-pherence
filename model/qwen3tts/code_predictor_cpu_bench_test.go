package qwen3tts

import (
	"os"
	"path/filepath"
	"testing"
)

// BenchmarkCodePredictorCPUReleasedFrame measures one acoustic frame using
// already-loaded, hash-verified 0.6B CustomVoice weights and a fixed owned
// hidden row. The Talker prefill and weight load are outside the timer.
func BenchmarkCodePredictorCPUReleasedFrame(b *testing.B) {
	dir := os.Getenv("GO_PHERENCE_QWEN3TTS_0B6_CUSTOMVOICE_DIR")
	if dir == "" {
		b.Skip("set GO_PHERENCE_QWEN3TTS_0B6_CUSTOMVOICE_DIR to pinned CustomVoice checkpoint")
	}
	b.StopTimer()
	for _, file := range []struct {
		path, hash string
		size       int64
	}{
		{filepath.Join(dir, "model.safetensors"), "bc3c7e785eb961179c25450d1acff03f839e0002f2f3a5aeb67b5735c0fa2adb", 1811626576},
		{filepath.Join(dir, "config.json"), "81aca2b6fac304944d8acf345272d8a9a727d5fc2e2e66b222ab4729340c7455", 0},
	} {
		if err := verifyReleasedFile(file.path, file.hash, file.size); err != nil {
			b.Fatal(err)
		}
	}
	cfg, err := ReadModelDir(dir)
	if err != nil {
		b.Fatal(err)
	}
	talker, err := LoadTalkerCPUFromDir(dir, cfg)
	if err != nil {
		b.Fatal(err)
	}
	predictor, err := LoadCodePredictorCPUFromDir(dir, cfg)
	if err != nil {
		b.Fatal(err)
	}
	plan, err := NewRuntimeRequestPlan(cfg, RuntimeRequest{Conditioning: ConditioningRequest{Speaker: Ryan, Language: English}, Prompt: PromptIDs{Text: []uint32{151644, 77091, 198, 151671, 151671, 151671, 151671, 151671, 151672, 9707, 1879}, Codec: []uint32{2154, 2156, 2050, 2157, 3061, 2148, 2149}}, MaxFrames: 1})
	if err != nil {
		b.Fatal(err)
	}
	first, err := talker.Prefill(plan)
	if err != nil {
		b.Fatal(err)
	}
	if first.SemanticToken != 1995 {
		b.Fatalf("first semantic=%d", first.SemanticToken)
	}
	b.ReportAllocs()
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		codes, logits, err := predictor.FirstAcousticFrame(talker, first.Hidden, first.SemanticToken)
		if err != nil {
			b.Fatal(err)
		}
		if len(codes) != 15 || len(logits) != cfg.CPVocabSize {
			b.Fatalf("frame dims codes=%d logits=%d", len(codes), len(logits))
		}
	}
}
