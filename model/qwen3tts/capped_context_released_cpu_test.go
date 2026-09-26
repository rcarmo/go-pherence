package qwen3tts

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"
)

func TestCappedSeededCPUContextReleasedHiTwoFrames(t *testing.T) {
	root := os.Getenv("GO_PHERENCE_QWEN3TTS_0B6_CUSTOMVOICE_DIR")
	if root == "" {
		t.Skip("set GO_PHERENCE_QWEN3TTS_0B6_CUSTOMVOICE_DIR to the pinned CustomVoice checkpoint")
	}
	for path, sha := range map[string]string{
		"config.json":                        "81aca2b6fac304944d8acf345272d8a9a727d5fc2e2e66b222ab4729340c7455",
		"model.safetensors":                  "bc3c7e785eb961179c25450d1acff03f839e0002f2f3a5aeb67b5735c0fa2adb",
		"vocab.json":                         "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
		"merges.txt":                         "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
		"speech_tokenizer/config.json":       "ee65bb901c876664ab8707c487157aa1a6ee57c65969b28fb5ec9dc211e68167",
		"speech_tokenizer/model.safetensors": "836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258",
	} {
		if err := verifyReleasedFile(filepath.Join(root, path), sha, 0); err != nil {
			t.Fatal(err)
		}
	}
	cfg, err := ReadModelDir(root)
	if err != nil {
		t.Fatal(err)
	}
	tok, err := LoadTokenizer(root)
	if err != nil {
		t.Fatal(err)
	}
	prompt, err := BuildCustomVoicePrompt(tok, "Hi", Ryan, English)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := NewRuntimeRequestPlan(cfg, RuntimeRequest{Conditioning: ConditioningRequest{Speaker: Ryan, Language: English}, Prompt: prompt, MaxFrames: 2})
	if err != nil {
		t.Fatal(err)
	}
	talker, err := LoadTalkerCPUFromDir(root, cfg)
	if err != nil {
		t.Fatal(err)
	}
	predictor, err := LoadCodePredictorCPUFromDir(root, cfg)
	if err != nil {
		t.Fatal(err)
	}
	decoder, err := LoadDecoder12HzCPUFromDir(root)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if got, err := GenerateCappedSeededCPUContext(ctx, plan, talker, predictor, decoder, 42); !errors.Is(err, context.Canceled) || !reflect.DeepEqual(got, BoundedCPUResult{}) {
		t.Fatalf("pre-cancelled result=%+v err=%v", got, err)
	}
	expired, stop := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
	defer stop()
	if got, err := GenerateCappedSeededCPUContext(expired, plan, talker, predictor, decoder, 42); !errors.Is(err, context.DeadlineExceeded) || !reflect.DeepEqual(got, BoundedCPUResult{}) {
		t.Fatalf("expired result=%+v err=%v", got, err)
	}
	legacy, err := GenerateCappedSeededCPU(plan, talker, predictor, decoder, 42)
	if err != nil {
		t.Fatal(err)
	}
	got, err := GenerateCappedSeededCPUContext(context.Background(), plan, talker, predictor, decoder, 42)
	if err != nil {
		t.Fatal(err)
	}
	if got.StoppedAtEOS || !reflect.DeepEqual(got.Semantic, []uint32{1995, 215}) || len(got.Acoustic) != 30 || len(got.Waveform) != 3840 || !reflect.DeepEqual(got, legacy) {
		t.Fatal("context path changed pinned short result")
	}
	got.Waveform[0]++
	if reflect.DeepEqual(got.Waveform, legacy.Waveform) {
		t.Fatal("context output aliases legacy result")
	}
}
