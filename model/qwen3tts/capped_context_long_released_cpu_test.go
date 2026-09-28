package qwen3tts

import (
	"context"
	"encoding/binary"
	"errors"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"sync"
	"testing"
)

// cancelAtLongFrame is test-owned context state. Count only the checks at the
// start of each continuation frame, as distinguished by the active call site
// in the production generator. Avoid clocks and global mutation.
type cancelAtLongFrame struct {
	context.Context
	cancel context.CancelFunc
	mu     sync.Mutex
	checks int
}

func (c *cancelAtLongFrame) Err() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.checks++
	// Four checks precede 10 prefix positions * 28 Talker layers. The first
	// acoustic frame is complete before those layers. Each continuation frame
	// adds a frame check, 28 layer checks, a selection check and an acoustic
	// check. After 31 continuations, 32 acoustic frames are complete.
	const firstFrameCheck = 4 + 10*28 + 1
	const checksPerFrame = 1 + 28 + 1 + 1
	if c.checks == firstFrameCheck+31*checksPerFrame {
		c.cancel()
	}
	return c.Context.Err()
}
func (c *cancelAtLongFrame) count() int { c.mu.Lock(); defer c.mu.Unlock(); return c.checks }

func TestCappedSentenceSeed42CPUReleasedCancellationRecovery(t *testing.T) {
	root := os.Getenv("GO_PHERENCE_QWEN3TTS_0B6_CUSTOMVOICE_DIR")
	oracle := os.Getenv("GO_PHERENCE_QWEN3TTS_SENTENCE64_ORACLE_DIR")
	if root == "" || oracle == "" {
		t.Skip("set GO_PHERENCE_QWEN3TTS_0B6_CUSTOMVOICE_DIR and GO_PHERENCE_QWEN3TTS_SENTENCE64_ORACLE_DIR")
	}
	o, codes := sentence64Fixture(t)
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
	wavePath := filepath.Join(oracle, "sentence_64_waveform.f32le")
	if err := verifyReleasedFile(wavePath, sentence64WaveSHA, 64*1920*4); err != nil {
		t.Fatal(err)
	}
	cfg, err := ReadModelDir(root)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.TalkerNumHiddenLayers != 28 {
		t.Fatalf("unexpected Talker layers %d", cfg.TalkerNumHiddenLayers)
	}
	tok, err := LoadTokenizer(root)
	if err != nil {
		t.Fatal(err)
	}
	prompt, err := BuildCustomVoicePrompt(tok, o.Text, Ryan, English)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(prompt.Text[CustomVoiceFirstTextIndex:], o.TextIDs) {
		t.Fatal("tokenizer changed")
	}
	plan, err := NewRuntimeRequestPlan(cfg, RuntimeRequest{Conditioning: ConditioningRequest{Speaker: Ryan, Language: English}, Prompt: prompt, MaxFrames: 64})
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
	defer cancel()
	marked := &cancelAtLongFrame{Context: ctx, cancel: cancel}
	got, err := GenerateCappedSeededCPUContext(marked, plan, talker, predictor, decoder, 42)
	if !errors.Is(err, context.Canceled) || !reflect.DeepEqual(got, BoundedCPUResult{}) {
		t.Fatalf("long cancellation returned partial result err=%v", err)
	}
	wantChecks := 4 + 10*28 + 1 + 31*(1+28+1+1)
	if marked.count() != wantChecks {
		t.Fatalf("cancelled after %d checks want %d", marked.count(), wantChecks)
	}
	runtime.GC()
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	recovered, err := GenerateCappedSeededCPUContext(context.Background(), plan, talker, predictor, decoder, 42)
	if err != nil {
		t.Fatal(err)
	}
	if recovered.StoppedAtEOS || len(recovered.Semantic) != 64 || len(recovered.Acoustic) != 960 || len(recovered.Waveform) != 122880 {
		t.Fatal("recovery geometry or EOS changed")
	}
	for f := 0; f < 64; f++ {
		if recovered.Semantic[f] != binary.LittleEndian.Uint32(codes[f*64:]) {
			t.Fatalf("semantic frame %d", f)
		}
		for g := 0; g < 15; g++ {
			if recovered.Acoustic[f*15+g] != binary.LittleEndian.Uint32(codes[f*64+(g+1)*4:]) {
				t.Fatalf("acoustic frame %d group %d", f, g)
			}
		}
	}
	wave, err := os.ReadFile(wavePath)
	if err != nil {
		t.Fatal(err)
	}
	maxErr := 0.0
	for i, v := range recovered.Waveform {
		r := math.Float32frombits(binary.LittleEndian.Uint32(wave[i*4:]))
		if !finiteTalkerValue(v) || !finiteTalkerValue(r) {
			t.Fatalf("nonfinite sample %d", i)
		}
		maxErr = math.Max(maxErr, math.Abs(float64(v)-float64(r)))
	}
	if maxErr > 1.6e-6 {
		t.Fatalf("recovery waveform max_abs=%g >1.6e-6", maxErr)
	}
	runtime.GC()
	runtime.ReadMemStats(&after)
	t.Logf("cancelled after 32 complete frames, checks=%d; recovery frames=64 codes=1024 samples=122880 max_abs=%g heap_before=%d heap_after=%d", marked.count(), maxErr, before.HeapAlloc, after.HeapAlloc)
	runtime.KeepAlive(recovered)
	runtime.KeepAlive(talker)
	runtime.KeepAlive(predictor)
	runtime.KeepAlive(decoder)
}
