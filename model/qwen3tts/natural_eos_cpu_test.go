package qwen3tts

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"testing"
)

func TestCappedCPUFrameAndEOSBoundaries(t *testing.T) {
	cfg := tinyTalkerConfig()
	talker, err := LoadTalkerCPU(tinyTalkerSource(cfg), cfg)
	if err != nil {
		t.Fatal(err)
	}
	predictor, err := LoadCodePredictorCPU(tinyPredictorSource(cfg), cfg)
	if err != nil {
		t.Fatal(err)
	}
	dc := tinyDecoder12HzConfig()
	decoder, err := LoadDecoder12HzCPU(tinyDecoderSource(dc), dc)
	if err != nil {
		t.Fatal(err)
	}
	prompt := tinyTalkerPlan(t, cfg).Prompt
	for _, tc := range []struct {
		cap, frames int
		eos         bool
	}{{32, 32, false}, {33, 32, true}, {64, 33, true}, {64, 46, true}, {64, 63, true}, {64, 64, false}} {
		t.Run(fmt.Sprintf("cap%d_frames%d_eos%t", tc.cap, tc.frames, tc.eos), func(t *testing.T) {
			plan, err := NewRuntimeRequestPlan(cfg, RuntimeRequest{Conditioning: ConditioningRequest{Speaker: Ryan, Language: English}, Prompt: prompt, MaxFrames: tc.cap})
			if err != nil {
				t.Fatal(err)
			}
			selected := 0
			choose := func(_ []float32, eos uint32) (uint32, error) {
				if tc.eos && selected == tc.frames {
					selected++
					return eos, nil
				}
				selected++
				return 1, nil
			}
			got, err := generateCappedGreedyCPU(plan, talker, predictor, decoder, choose)
			if err != nil {
				t.Fatal(err)
			}
			if got.StoppedAtEOS != tc.eos || len(got.Semantic) != tc.frames || len(got.Acoustic) != tc.frames*15 || len(got.Waveform) != tc.frames*1920 || len(got.ContinuationHidden) != tc.frames-1 || len(got.ContinuationLogits) != tc.frames-1 {
				t.Fatal("frame/EOS geometry", got.StoppedAtEOS, len(got.Semantic), len(got.Waveform))
			}
			for _, id := range got.Semantic {
				if id == CodecEOS {
					t.Fatal("EOS emitted as frame")
				}
			}
		})
	}
}

// This released test requires both the approved checkpoint and regenerated,
// independently hash-verified Rust waveform. Missing assets skip explicitly.
func TestCappedHiSeed42CPUReleasedNaturalEOS(t *testing.T) {
	root := os.Getenv("GO_PHERENCE_QWEN3TTS_0B6_CUSTOMVOICE_DIR")
	oracle := os.Getenv("GO_PHERENCE_QWEN3TTS_HI64_ORACLE_DIR")
	if root == "" || oracle == "" {
		t.Skip("set GO_PHERENCE_QWEN3TTS_0B6_CUSTOMVOICE_DIR and GO_PHERENCE_QWEN3TTS_HI64_ORACLE_DIR")
	}
	fixture := filepath.Join("testdata", "customvoice_0b6_ryan_hello")
	files := map[string]string{
		filepath.Join(root, "config.json"):                           "81aca2b6fac304944d8acf345272d8a9a727d5fc2e2e66b222ab4729340c7455",
		filepath.Join(root, "model.safetensors"):                     "bc3c7e785eb961179c25450d1acff03f839e0002f2f3a5aeb67b5735c0fa2adb",
		filepath.Join(root, "vocab.json"):                            "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
		filepath.Join(root, "merges.txt"):                            "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
		filepath.Join(root, "speech_tokenizer", "config.json"):       "ee65bb901c876664ab8707c487157aa1a6ee57c65969b28fb5ec9dc211e68167",
		filepath.Join(root, "speech_tokenizer", "model.safetensors"): "836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258",
		filepath.Join(oracle, "probe_hi_64_waveform.f32le"):          "8011f727acf104e24e606b5aea4f3881bf55e1d5d575dd5063586a72dce33da2",
	}
	for file, want := range files {
		if err := verifyReleasedFile(file, want, 0); err != nil {
			t.Fatal(err)
		}
	}
	TestPinnedHi64FrameEOSObservationFixture(t)
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
	runtime.GC()
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	got, err := GenerateCappedSeededCPU(plan, talker, predictor, decoder, 42)
	if err != nil {
		t.Fatal(err)
	}
	if !got.StoppedAtEOS || len(got.Semantic) != 46 || len(got.Acoustic) != 690 || len(got.Waveform) != 88320 || len(got.ContinuationHidden) != 45 || len(got.ContinuationLogits) != 45 {
		t.Fatalf("EOS geometry stopped=%t semantic=%d acoustic=%d samples=%d", got.StoppedAtEOS, len(got.Semantic), len(got.Acoustic), len(got.Waveform))
	}
	codes, err := os.ReadFile(filepath.Join(fixture, "probe_hi_64_codes.u32le"))
	if err != nil {
		t.Fatal(err)
	}
	for frame := 0; frame < 46; frame++ {
		if got.Semantic[frame] != binary.LittleEndian.Uint32(codes[(frame*16)*4:]) {
			t.Fatalf("semantic mismatch frame%d", frame)
		}
		for group := 0; group < 15; group++ {
			if got.Acoustic[frame*15+group] != binary.LittleEndian.Uint32(codes[(frame*16+group+1)*4:]) {
				t.Fatalf("acoustic mismatch frame%d group%d", frame, group)
			}
		}
	}
	wantWave, err := os.ReadFile(filepath.Join(oracle, "probe_hi_64_waveform.f32le"))
	if err != nil || len(wantWave) != len(got.Waveform)*4 {
		t.Fatal("reference waveform geometry", err)
	}
	var maxErr float64
	for i, v := range got.Waveform {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("nonfinite%d", i)
		}
		maxErr = math.Max(maxErr, math.Abs(float64(v)-float64(math.Float32frombits(binary.LittleEndian.Uint32(wantWave[i*4:])))))
	}
	if maxErr > 1.6e-6 {
		t.Fatalf("naturalEOSwave error%g >1.6e-6", maxErr)
	}
	heldWave := append([]float32(nil), got.Waveform...)
	heldCodes := append([]uint32(nil), got.Semantic...)
	// Same model instance still reproduces the already-qualified cap32 prefix.
	shortPlan, err := NewRuntimeRequestPlan(cfg, RuntimeRequest{Conditioning: ConditioningRequest{Speaker: Ryan, Language: English}, Prompt: prompt, MaxFrames: 32})
	if err != nil {
		t.Fatal(err)
	}
	short, err := GenerateCappedSeededCPU(shortPlan, talker, predictor, decoder, 42)
	if err != nil {
		t.Fatal(err)
	}
	if short.StoppedAtEOS || !reflect.DeepEqual(short.Semantic, got.Semantic[:32]) || !reflect.DeepEqual(short.Acoustic, got.Acoustic[:32*15]) {
		t.Fatal("cap32 prefix drift")
	}
	if !reflect.DeepEqual(got.Waveform, heldWave) || !reflect.DeepEqual(got.Semantic, heldCodes) {
		t.Fatal("retained output mutated")
	}
	runtime.GC()
	runtime.ReadMemStats(&after)
	t.Logf("NATURAL_EOS frames=46 step=46 codes=736 samples=88320 waveform_max_abs=%g heap_before=%d heap_after=%d", maxErr, before.HeapAlloc, after.HeapAlloc)
	if path := os.Getenv("GO_PHERENCE_QWEN3TTS_HI64_OUTPUT"); path != "" {
		data := make([]byte, len(got.Waveform)*4)
		for i, v := range got.Waveform {
			binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(v))
		}
		if err = os.WriteFile(path, data, 0600); err != nil {
			t.Fatal(err)
		}
	}
	runtime.KeepAlive(talker)
	runtime.KeepAlive(predictor)
	runtime.KeepAlive(decoder)
	runtime.KeepAlive(got)
	runtime.KeepAlive(short)
}
