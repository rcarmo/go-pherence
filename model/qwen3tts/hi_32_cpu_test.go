package qwen3tts

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestCappedHiSeed42CPUReleasedThirtyTwoFrames(t *testing.T) {
	const envName = "GO_PHERENCE_QWEN3TTS_0B6_CUSTOMVOICE_DIR"
	dir := os.Getenv(envName)
	if dir == "" {
		t.Skipf("set %s to pinned CustomVoice checkpoint", envName)
	}
	root := filepath.Join("testdata", "customvoice_0b6_ryan_hello")
	data, err := os.ReadFile(filepath.Join(root, "reference.json"))
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		ModelRevision      string  `json:"model_revision"`
		OracleRevision     string  `json:"oracle_revision"`
		ModelSHA           string  `json:"model_safetensors_sha256"`
		ModelSize          int64   `json:"model_safetensors_size"`
		ConfigSHA          string  `json:"model_config_sha256"`
		TokenizerSHA       string  `json:"speech_tokenizer_model_sha256"`
		TokenizerSize      int64   `json:"speech_tokenizer_model_size"`
		TokenizerConfigSHA string  `json:"speech_tokenizer_config_sha256"`
		ScriptSHA          string  `json:"hi_32_eos_probe_script_sha256"`
		CodesSHA           string  `json:"hi_32_eos_probe_codes_sha256"`
		WaveSHA            string  `json:"hi_32_eos_probe_waveform_sha256"`
		Frames             int     `json:"hi_32_eos_probe_frames"`
		Seed               uint64  `json:"hi_32_eos_probe_seed"`
		EOS                bool    `json:"hi_32_eos_probe_eos_observed"`
		FirstToken         uint32  `json:"hi_seeded_sixteen_first_text_token"`
		MaxAbs             float64 `json:"hi_32_go_waveform_max_abs_threshold"`
	}
	if err := json.Unmarshal(data, &ref); err != nil {
		t.Fatal(err)
	}
	if ref.ModelRevision != "85e237c12c027371202489a0ec509ded67b5e4b5" || ref.OracleRevision != "711ceee07cad92673f86de8997bdf54c30caa49f" || ref.ModelSHA != "bc3c7e785eb961179c25450d1acff03f839e0002f2f3a5aeb67b5735c0fa2adb" || ref.ModelSize != 1811626576 || ref.ConfigSHA != "81aca2b6fac304944d8acf345272d8a9a727d5fc2e2e66b222ab4729340c7455" || ref.TokenizerSHA != "836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258" || ref.TokenizerSize != 682293092 || ref.TokenizerConfigSHA != "ee65bb901c876664ab8707c487157aa1a6ee57c65969b28fb5ec9dc211e68167" || ref.ScriptSHA != "8acce5b9e8fb6cae755f23ae019416329a26b9fad1529f773bd0043e5e1b91b6" || ref.CodesSHA != "88aafb544a81f0f09d4b83125c9b4f01370ade24fe24f47e9fcde86818a58391" || ref.WaveSHA != "831831b02d90b880311275807008585f82a9425f32db7401a4c6996397039a82" || ref.Frames != 32 || ref.Seed != 42 || ref.EOS || ref.FirstToken != 13048 || ref.MaxAbs != 1.6e-6 {
		t.Fatal("unexpected pinned 32-frame provenance")
	}
	for _, f := range []struct {
		path, hash string
		size       int64
	}{
		{filepath.Join("..", "..", "scripts", "qwen3tts_probe_hi_32_eos.rs"), ref.ScriptSHA, 0},
		{filepath.Join(root, "probe_hi_32_codes.u32le"), ref.CodesSHA, 32 * 16 * 4},
		{filepath.Join(dir, "model.safetensors"), ref.ModelSHA, ref.ModelSize},
		{filepath.Join(dir, "config.json"), ref.ConfigSHA, 0},
		{filepath.Join(dir, "speech_tokenizer", "model.safetensors"), ref.TokenizerSHA, ref.TokenizerSize},
		{filepath.Join(dir, "speech_tokenizer", "config.json"), ref.TokenizerConfigSHA, 0},
	} {
		if err := verifyReleasedFile(f.path, f.hash, f.size); err != nil {
			t.Fatal(err)
		}
	}
	wavePath := filepath.Join(root, "probe_hi_32_waveform.f32le")
	if err := verifyReleasedFile(wavePath, ref.WaveSHA, 32*1920*4); err != nil {
		t.Fatal(err)
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
	prompt, err := BuildCustomVoicePrompt(tok, "Hi", Ryan, English)
	if err != nil {
		t.Fatal(err)
	}
	if len(prompt.Text) != 10 || prompt.Text[9] != ref.FirstToken {
		t.Fatalf("Hi tokenization=%v", prompt.Text)
	}
	plan, err := NewRuntimeRequestPlan(cfg, RuntimeRequest{Conditioning: ConditioningRequest{Speaker: Ryan, Language: English}, Prompt: prompt, MaxFrames: ref.Frames})
	if err != nil {
		t.Fatal(err)
	}
	result, err := GenerateCappedSeededCPU(plan, talker, predictor, decoder, ref.Seed)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Semantic) != 32 || len(result.Acoustic) != 480 || len(result.Waveform) != 61440 {
		t.Fatalf("32-frame result geometry semantic=%d acoustic=%d samples=%d", len(result.Semantic), len(result.Acoustic), len(result.Waveform))
	}
	codes, err := os.ReadFile(filepath.Join(root, "probe_hi_32_codes.u32le"))
	if err != nil {
		t.Fatal(err)
	}
	for frame := 0; frame < 32; frame++ {
		if want := binary.LittleEndian.Uint32(codes[frame*64:]); result.Semantic[frame] != want || want == CodecEOS {
			t.Fatalf("semantic frame=%d got=%d want=%d", frame, result.Semantic[frame], want)
		}
		for group := 0; group < 15; group++ {
			want := binary.LittleEndian.Uint32(codes[frame*64+(group+1)*4:])
			if result.Acoustic[frame*15+group] != want {
				t.Fatalf("acoustic frame=%d group=%d got=%d want=%d", frame, group, result.Acoustic[frame*15+group], want)
			}
		}
	}
	wave, err := os.ReadFile(wavePath)
	if err != nil {
		t.Fatal(err)
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
	if maxAbs > ref.MaxAbs {
		t.Fatalf("32-frame waveform max_abs=%g threshold=%g", maxAbs, ref.MaxAbs)
	}
	if !reflect.DeepEqual(result.Semantic[:16], []uint32{1995, 215, 212, 1181, 462, 251, 122, 122, 122, 1738, 1738, 1738, 1738, 1738, 1738, 1738}) {
		t.Fatal("16-frame prefix changed")
	}
	t.Logf("32-frame seed=%d samples=%d max_abs=%g threshold=%g", ref.Seed, len(result.Waveform), maxAbs, ref.MaxAbs)
}
