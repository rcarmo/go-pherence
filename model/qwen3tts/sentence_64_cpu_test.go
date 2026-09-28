package qwen3tts

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"testing"
)

const sentence64Text = "The quick brown fox jumps over the lazy dog and then walks slowly back to the house, where a warm dinner is waiting."
const sentence64WaveSHA = "e0ef875fbe94c1eb923b78c28fc72e5bca87af0b28985db07d3e8e641dcb02fb"

type sentence64Observation struct {
	Text                 string `json:"text"`
	Speaker              string `json:"speaker"`
	Language             string `json:"language"`
	Seed                 uint64 `json:"seed"`
	Cap, Frames, Samples int
	EOSStep              *int     `json:"eos_step"`
	EOSToken             uint32   `json:"eos_token_id"`
	TextIDs              []uint32 `json:"text_ids"`
}

func sentence64Fixture(t *testing.T) (sentence64Observation, []byte) {
	t.Helper()
	root := filepath.Join("testdata", "customvoice_0b6_ryan_hello")
	for _, f := range []struct {
		path, sha string
		size      int64
	}{
		{filepath.Join("..", "..", "scripts", "qwen3tts_probe_sentence_64.rs"), "2684e2892e49b2ed9e42adbe20b5d4afd04e86f90c95f5e15e2918cb5638a9d0", 0},
		{filepath.Join(root, "sentence_64_observation.json"), "874709a292ba4659d0b0354be9a0893e737dc366f12ebc0e55d3eb0c825bb994", 0},
		{filepath.Join(root, "sentence_64_codes.u32le"), "fe1f4c0dd8f99eb2dfeb1f9528a015142fdac186f0bd1237eaf0506d41a003cf", 64 * 16 * 4},
	} {
		if err := verifyReleasedFile(f.path, f.sha, f.size); err != nil {
			t.Fatal(err)
		}
	}
	data, err := os.ReadFile(filepath.Join(root, "sentence_64_observation.json"))
	if err != nil {
		t.Fatal(err)
	}
	var o sentence64Observation
	if err = json.Unmarshal(data, &o); err != nil {
		t.Fatal(err)
	}
	if o.Text != sentence64Text || o.Speaker != "ryan" || o.Language != "en" || o.Seed != 42 || o.Cap != 64 || o.Frames != 64 || o.EOSStep != nil || o.EOSToken != CodecEOS || o.Samples != 122880 || len(o.TextIDs) != 25 {
		t.Fatal("unexpected cap64 provenance")
	}
	codes, err := os.ReadFile(filepath.Join(root, "sentence_64_codes.u32le"))
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 64*16; i++ {
		id := binary.LittleEndian.Uint32(codes[i*4:])
		if i%16 == 0 {
			if id >= 2048 || id == CodecEOS {
				t.Fatal("semantic domain", i, id)
			}
		} else if id >= 2048 {
			t.Fatal("acoustic domain", i, id)
		}
	}
	return o, codes
}

func TestPinnedSentence64Fixture(t *testing.T) { sentence64Fixture(t) }

func TestCappedSentenceSeed42CPUReleasedSixtyFourFrames(t *testing.T) {
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
	tok, err := LoadTokenizer(root)
	if err != nil {
		t.Fatal(err)
	}
	prompt, err := BuildCustomVoicePrompt(tok, o.Text, Ryan, English)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(prompt.Text[CustomVoiceFirstTextIndex:], o.TextIDs) {
		t.Fatal("independent tokenization mismatch", prompt.Text)
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
	if got.StoppedAtEOS || len(got.Semantic) != 64 || len(got.Acoustic) != 960 || len(got.Waveform) != 122880 || len(got.ContinuationHidden) != 63 || len(got.ContinuationLogits) != 63 {
		t.Fatalf("cap64 geometry EOS=%t frames=%d acoustic=%d samples=%d", got.StoppedAtEOS, len(got.Semantic), len(got.Acoustic), len(got.Waveform))
	}
	for i := 0; i < 64; i++ {
		if got.Semantic[i] != binary.LittleEndian.Uint32(codes[i*64:]) {
			t.Fatalf("semantic frame%d got%d want%d", i, got.Semantic[i], binary.LittleEndian.Uint32(codes[i*64:]))
		}
		for g := 0; g < 15; g++ {
			if got.Acoustic[i*15+g] != binary.LittleEndian.Uint32(codes[i*64+(g+1)*4:]) {
				t.Fatalf("acoustic frame%d group%d", i, g)
			}
		}
	}
	want, err := os.ReadFile(wavePath)
	if err != nil {
		t.Fatal(err)
	}
	maxErr := 0.0
	for i, v := range got.Waveform {
		r := math.Float32frombits(binary.LittleEndian.Uint32(want[i*4:]))
		if !finiteTalkerValue(v) || !finiteTalkerValue(r) {
			t.Fatalf("nonfinite%d", i)
		}
		maxErr = math.Max(maxErr, math.Abs(float64(v)-float64(r)))
	}
	if maxErr > 1.6e-6 {
		t.Fatalf("cap64 waveform max_abs=%g >1.6e-6", maxErr)
	}
	// Reuse with a different short prompt; the expensive EOS result is already
	// independently qualified, so only a small mixed-input recovery is needed here.
	savedWave := append([]float32(nil), got.Waveform...)
	savedSemantic := append([]uint32(nil), got.Semantic...)
	hi, err := BuildCustomVoicePrompt(tok, "Hi", Ryan, English)
	if err != nil {
		t.Fatal(err)
	}
	shortPlan, err := NewRuntimeRequestPlan(cfg, RuntimeRequest{Conditioning: ConditioningRequest{Speaker: Ryan, Language: English}, Prompt: hi, MaxFrames: 2})
	if err != nil {
		t.Fatal(err)
	}
	short, err := GenerateCappedSeededCPU(shortPlan, talker, predictor, decoder, 42)
	if err != nil {
		t.Fatal(err)
	}
	if short.StoppedAtEOS || !reflect.DeepEqual(short.Semantic, []uint32{1995, 215}) || len(short.Acoustic) != 30 || len(short.Waveform) != 3840 {
		t.Fatal("mixed-input recovery")
	}
	if !reflect.DeepEqual(got.Waveform, savedWave) || !reflect.DeepEqual(got.Semantic, savedSemantic) {
		t.Fatal("retained output mutated")
	}
	runtime.GC()
	runtime.ReadMemStats(&after)
	t.Logf("CAP64 EOS=false frames=64 codes=1024 samples=122880 max_abs=%g heap_before=%d heap_after=%d allocated_bytes=%d allocations=%d", maxErr, before.HeapAlloc, after.HeapAlloc, after.TotalAlloc-before.TotalAlloc, after.Mallocs-before.Mallocs)
	if path := os.Getenv("GO_PHERENCE_QWEN3TTS_SENTENCE64_OUTPUT"); path != "" {
		data := make([]byte, len(got.Waveform)*4)
		for i, v := range got.Waveform {
			binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(v))
		}
		if err := os.WriteFile(path, data, 0600); err != nil {
			t.Fatal(err)
		}
	}
	runtime.KeepAlive(got)
	runtime.KeepAlive(short)
	runtime.KeepAlive(talker)
	runtime.KeepAlive(predictor)
	runtime.KeepAlive(decoder)
}
