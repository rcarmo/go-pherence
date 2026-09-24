package qwen3tts

import (
	"os"
	"path/filepath"
	"testing"
)

// The released benchmarks measure synthesis only; all weight loads and prompt
// construction are outside the timed region. Both require the pinned model.
func BenchmarkCappedGreedyCPUReleasedEightFrames(b *testing.B) {
	benchmarkCappedGreedyCPUReleased(b, 8)
}
func BenchmarkCappedGreedyCPUReleasedSixteenFrames(b *testing.B) {
	benchmarkCappedGreedyCPUReleased(b, 16)
}

func benchmarkCappedGreedyCPUReleased(b *testing.B, frames int) {
	b.Helper()
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
		{filepath.Join(dir, "speech_tokenizer", "model.safetensors"), "836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258", 682293092},
		{filepath.Join(dir, "speech_tokenizer", "config.json"), "ee65bb901c876664ab8707c487157aa1a6ee57c65969b28fb5ec9dc211e68167", 0},
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
	decoder, err := LoadDecoder12HzCPUFromDir(dir)
	if err != nil {
		b.Fatal(err)
	}
	tok, err := LoadTokenizer(dir)
	if err != nil {
		b.Fatal(err)
	}
	prompt, err := BuildCustomVoicePrompt(tok, "Hello world", Ryan, English)
	if err != nil {
		b.Fatal(err)
	}
	plan, err := NewRuntimeRequestPlan(cfg, RuntimeRequest{Conditioning: ConditioningRequest{Speaker: Ryan, Language: English}, Prompt: prompt, MaxFrames: frames})
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		result, err := GenerateCappedGreedyCPU(plan, talker, predictor, decoder)
		if err != nil {
			b.Fatal(err)
		}
		if len(result.Semantic) != frames || len(result.Waveform) != frames*1920 {
			b.Fatalf("unexpected %d-frame output %d/%d", frames, len(result.Semantic), len(result.Waveform))
		}
	}
}
