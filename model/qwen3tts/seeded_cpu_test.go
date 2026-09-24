package qwen3tts

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"sync"
	"testing"
)

func TestCappedSeededCPUSyntheticOwnership(t *testing.T) {
	cfg := tinyTalkerConfig()
	talker, err := LoadTalkerCPU(tinyTalkerSource(cfg), cfg)
	if err != nil {
		t.Fatal(err)
	}
	predictor, err := LoadCodePredictorCPU(tinyPredictorSource(cfg), cfg)
	if err != nil {
		t.Fatal(err)
	}
	decoder, err := LoadDecoder12HzCPU(tinyDecoderSource(tinyDecoder12HzConfig()), tinyDecoder12HzConfig())
	if err != nil {
		t.Fatal(err)
	}
	plan := tinyTalkerPlan(t, cfg)
	want, err := GenerateCappedSeededCPU(plan, talker, predictor, decoder, 42)
	if err != nil {
		t.Fatal(err)
	}
	if len(want.Semantic) != 2 || len(want.Acoustic) != 30 || len(want.Waveform) != 3840 {
		t.Fatalf("unexpected geometry %+v", want.Semantic)
	}
	const workers = 8
	results := make([]BoundedCPUResult, workers)
	errs := make([]error, workers)
	var wg sync.WaitGroup
	for i := range results {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			results[i], errs[i] = GenerateCappedSeededCPU(plan, talker, predictor, decoder, 42)
		}(i)
	}
	wg.Wait()
	for i, r := range results {
		if errs[i] != nil || !reflect.DeepEqual(r, want) {
			t.Fatalf("request %d differs: %v", i, errs[i])
		}
	}
	results[0].Waveform[0]++
	if results[0].Waveform[0] == results[1].Waveform[0] {
		t.Fatal("request outputs alias")
	}
	bad := plan
	bad.MaxFrames = 33
	if got, err := GenerateCappedSeededCPU(bad, talker, predictor, decoder, 42); err == nil || got.Waveform != nil {
		t.Fatal("accepted over-cap request")
	}
}

func TestCappedSeededCPUReleasedSixteenFrames(t *testing.T) {
	testCappedSeededCPUReleasedSixteenFrames(t, 42)
}
func TestCappedSeed7CPUReleasedSixteenFrames(t *testing.T) {
	testCappedSeededCPUReleasedSixteenFrames(t, 7)
}
func TestCappedHiSeededCPUReleasedSixteenFrames(t *testing.T) {
	testCappedSeededCPUReleasedSixteenFrames(t, 42, "Hi")
}

func testCappedSeededCPUReleasedSixteenFrames(t *testing.T, seed uint64, textOverride ...string) {
	t.Helper()
	const envName = "GO_PHERENCE_QWEN3TTS_0B6_CUSTOMVOICE_DIR"
	dir := os.Getenv(envName)
	if dir == "" {
		t.Skipf("set %s to pinned Qwen3-TTS 0.6B CustomVoice directory", envName)
	}
	root := filepath.Join("testdata", "customvoice_0b6_ryan_hello")
	data, err := os.ReadFile(filepath.Join(root, "reference.json"))
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		ModelRevision               string   `json:"model_revision"`
		OracleRevision              string   `json:"oracle_revision"`
		ModelSafetensorsSHA256      string   `json:"model_safetensors_sha256"`
		ModelSafetensorsSize        int64    `json:"model_safetensors_size"`
		ModelConfigSHA256           string   `json:"model_config_sha256"`
		SpeechTokenizerModelSHA256  string   `json:"speech_tokenizer_model_sha256"`
		SpeechTokenizerModelSize    int64    `json:"speech_tokenizer_model_size"`
		SpeechTokenizerConfigSHA256 string   `json:"speech_tokenizer_config_sha256"`
		ScriptSHA                   string   `json:"seeded_sixteen_oracle_script_sha256"`
		Seed                        uint64   `json:"seeded_sixteen_seed"`
		Semantic                    []uint32 `json:"seeded_sixteen_semantic"`
		CodesSHA                    string   `json:"seeded_sixteen_codes_sha256"`
		WaveformSHA                 string   `json:"seeded_sixteen_waveform_sha256"`
		Threshold                   float64  `json:"seeded_sixteen_waveform_max_abs_threshold"`
		Seed7ScriptSHA              string   `json:"seed7_sixteen_oracle_script_sha256"`
		Seed7Seed                   uint64   `json:"seed7_sixteen_seed"`
		Seed7Semantic               []uint32 `json:"seed7_sixteen_semantic"`
		Seed7CodesSHA               string   `json:"seed7_sixteen_codes_sha256"`
		Seed7WaveformSHA            string   `json:"seed7_sixteen_waveform_sha256"`
		Seed7Threshold              float64  `json:"seed7_sixteen_waveform_max_abs_threshold"`
		HiScriptSHA                 string   `json:"hi_seeded_sixteen_oracle_script_sha256"`
		HiFirstTextToken            uint32   `json:"hi_seeded_sixteen_first_text_token"`
		HiSeed                      uint64   `json:"hi_seeded_sixteen_seed"`
		HiSemantic                  []uint32 `json:"hi_seeded_sixteen_semantic"`
		HiCodesSHA                  string   `json:"hi_seeded_sixteen_codes_sha256"`
		HiWaveformSHA               string   `json:"hi_seeded_sixteen_waveform_sha256"`
		HiThreshold                 float64  `json:"hi_seeded_sixteen_waveform_max_abs_threshold"`
	}
	if err := json.Unmarshal(data, &ref); err != nil {
		t.Fatal(err)
	}
	if ref.ModelRevision != "85e237c12c027371202489a0ec509ded67b5e4b5" || ref.OracleRevision != "711ceee07cad92673f86de8997bdf54c30caa49f" || ref.ModelSafetensorsSHA256 != "bc3c7e785eb961179c25450d1acff03f839e0002f2f3a5aeb67b5735c0fa2adb" || ref.ModelSafetensorsSize != 1811626576 || ref.ModelConfigSHA256 != "81aca2b6fac304944d8acf345272d8a9a727d5fc2e2e66b222ab4729340c7455" || ref.SpeechTokenizerModelSHA256 != "836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258" || ref.SpeechTokenizerModelSize != 682293092 || ref.SpeechTokenizerConfigSHA256 != "ee65bb901c876664ab8707c487157aa1a6ee57c65969b28fb5ec9dc211e68167" || ref.ScriptSHA != "1ad6801a68629e235497efa3ae8fc96709421504cbd313309a93f434ffab6386" || ref.Seed != 42 || !reflect.DeepEqual(ref.Semantic, []uint32{1995, 215, 212, 1181, 462, 251, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122}) || ref.CodesSHA != "fcfb2aab18463a44ef431ead7bd1eb8aff4aa13040103cf67e077c2749fe57a8" || ref.WaveformSHA != "18e69b94c9e06035a0a3b71d5a36c0166d1b2765b8691440f0b3159fc4be817b" || ref.Threshold != 6.5e-6 {
		t.Fatal("unexpected pinned seeded sixteen-frame provenance")
	}
	if seed != 42 && seed != 7 {
		t.Fatalf("unsupported seeded fixture %d", seed)
	}
	text := "Hello world"
	if len(textOverride) > 0 {
		text = textOverride[0]
		if text != "Hi" || seed != 42 {
			t.Fatalf("unsupported prompt/seed %q/%d", text, seed)
		}
	}
	prefix, script := "seeded_sixteen", "qwen3tts_oracle_seeded_sixteen_frames.rs"
	semantic, codesSHA, waveformSHA, scriptSHA, threshold := ref.Semantic, ref.CodesSHA, ref.WaveformSHA, ref.ScriptSHA, ref.Threshold
	if text == "Hi" {
		if ref.HiScriptSHA != "a28aac5b628164f01176773b753a1bbbfbf5efce56d9c01b2fa1e1666fb002ee" || ref.HiFirstTextToken != 13048 || ref.HiSeed != 42 || !reflect.DeepEqual(ref.HiSemantic, []uint32{1995, 215, 212, 1181, 462, 251, 122, 122, 122, 1738, 1738, 1738, 1738, 1738, 1738, 1738}) || ref.HiCodesSHA != "17b16ec184d5a075805c057aa5eaa91b1960f97710f0f06206c6676f8b861a52" || ref.HiWaveformSHA != "373f8c3f6f0f9b2e39a81487584160fbf183813001ea303dfb0869c816df740f" || ref.HiThreshold != 1.6e-6 {
			t.Fatal("unexpected pinned Hi provenance")
		}
		prefix, script = "hi_seeded_sixteen", "qwen3tts_oracle_hi_seeded_sixteen.rs"
		semantic, codesSHA, waveformSHA, scriptSHA, threshold = ref.HiSemantic, ref.HiCodesSHA, ref.HiWaveformSHA, ref.HiScriptSHA, ref.HiThreshold
	} else if seed == 7 {
		if ref.Seed7ScriptSHA != "cf9639ee01098a07516d9f6599abec6154fa3cd9c213e58a4bfc44dfdaa46406" || ref.Seed7Seed != 7 || !reflect.DeepEqual(ref.Seed7Semantic, []uint32{1995, 215, 212, 1181, 462, 1225, 1029, 122, 122, 1891, 1891, 1891, 1891, 1891, 1891, 1891}) || ref.Seed7CodesSHA != "3d955a9dd5074d8be14976d0c53afbb0704b52f58ade0584b8bfe740abc9a92d" || ref.Seed7WaveformSHA != "01de12600f88be6964a76bdbe0e99eed925c4b4b1aa75444a1bd934a83ff9a06" || ref.Seed7Threshold != 3.5e-6 {
			t.Fatal("unexpected pinned seed-seven provenance")
		}
		prefix, script = "seed7_sixteen", "qwen3tts_oracle_seed7_sixteen_frames.rs"
		semantic, codesSHA, waveformSHA, scriptSHA, threshold = ref.Seed7Semantic, ref.Seed7CodesSHA, ref.Seed7WaveformSHA, ref.Seed7ScriptSHA, ref.Seed7Threshold
	}
	for _, file := range []struct {
		path, hash string
		size       int64
	}{
		{filepath.Join("..", "..", "scripts", script), scriptSHA, 0},
		{filepath.Join(dir, "model.safetensors"), ref.ModelSafetensorsSHA256, ref.ModelSafetensorsSize},
		{filepath.Join(dir, "config.json"), ref.ModelConfigSHA256, 0},
		{filepath.Join(dir, "speech_tokenizer", "model.safetensors"), ref.SpeechTokenizerModelSHA256, ref.SpeechTokenizerModelSize},
		{filepath.Join(dir, "speech_tokenizer", "config.json"), ref.SpeechTokenizerConfigSHA256, 0},
	} {
		if err := verifyReleasedFile(file.path, file.hash, file.size); err != nil {
			t.Fatal(err)
		}
	}
	cfg, err := ReadModelDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	talker, err := LoadTalkerCPUFromDir(dir, cfg)
	if err != nil {
		t.Fatal(err)
	}
	predictor, err := LoadCodePredictorCPUFromDir(dir, cfg)
	if err != nil {
		t.Fatal(err)
	}
	decoder, err := LoadDecoder12HzCPUFromDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	tok, err := LoadTokenizer(dir)
	if err != nil {
		t.Fatal(err)
	}
	prompt, err := BuildCustomVoicePrompt(tok, text, Ryan, English)
	if err != nil {
		t.Fatal(err)
	}
	if text == "Hi" && (len(prompt.Text) != 10 || prompt.Text[9] != ref.HiFirstTextToken) {
		t.Fatalf("Hi tokenization mismatch %v", prompt.Text)
	}
	plan, err := NewRuntimeRequestPlan(cfg, RuntimeRequest{Conditioning: ConditioningRequest{Speaker: Ryan, Language: English}, Prompt: prompt, MaxFrames: 16})
	if err != nil {
		t.Fatal(err)
	}
	result, err := GenerateCappedSeededCPU(plan, talker, predictor, decoder, seed)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(result.Semantic, semantic) || len(result.Acoustic) != 240 || len(result.Waveform) != 30720 {
		t.Fatalf("seeded geometry semantic=%v acoustic=%d samples=%d", result.Semantic, len(result.Acoustic), len(result.Waveform))
	}
	codes, err := os.ReadFile(filepath.Join(root, prefix+"_codes.u32le"))
	if err != nil {
		t.Fatal(err)
	}
	if len(codes) != 16*16*4 || hashCP(codes) != codesSHA {
		t.Fatal("invalid seeded code fixture")
	}
	for frame := 0; frame < 16; frame++ {
		if binary.LittleEndian.Uint32(codes[frame*64:]) != result.Semantic[frame] {
			t.Fatalf("semantic frame %d", frame)
		}
		for group := 0; group < 15; group++ {
			want := binary.LittleEndian.Uint32(codes[frame*64+(group+1)*4:])
			if result.Acoustic[frame*15+group] != want {
				t.Fatalf("acoustic frame=%d group=%d got=%d want=%d", frame, group, result.Acoustic[frame*15+group], want)
			}
		}
	}
	wave, err := os.ReadFile(filepath.Join(root, prefix+"_waveform.f32le"))
	if err != nil {
		t.Fatal(err)
	}
	if len(wave) != len(result.Waveform)*4 || hashCP(wave) != waveformSHA {
		t.Fatal("invalid seeded waveform fixture")
	}
	var maxAbs float64
	for i, v := range result.Waveform {
		want := math.Float32frombits(binary.LittleEndian.Uint32(wave[i*4:]))
		if !finiteTalkerValue(v) || !finiteTalkerValue(want) {
			t.Fatalf("nonfinite sample %d", i)
		}
		d := math.Abs(float64(v) - float64(want))
		if d > maxAbs {
			maxAbs = d
		}
	}
	if maxAbs > threshold {
		t.Fatalf("seed=%d waveform max_abs=%g threshold=%g", seed, maxAbs, threshold)
	}
	t.Logf("seed=%d sixteen-frame samples=%d max_abs=%g threshold=%g", seed, len(result.Waveform), maxAbs, threshold)
}
