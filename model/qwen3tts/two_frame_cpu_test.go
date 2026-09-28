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

func TestTwoFrameCPURejectsMalformed(t *testing.T) {
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
	for _, tc := range []struct {
		name      string
		talker    *TalkerCPU
		predictor *CodePredictorCPU
		decoder   *Decoder12HzCPU
		plan      RuntimeRequestPlan
	}{
		{"nil talker", nil, predictor, decoder, plan},
		{"nil predictor", talker, nil, decoder, plan},
		{"nil decoder", talker, predictor, nil, plan},
		{"wrong frame budget", talker, predictor, decoder, func() RuntimeRequestPlan { p := plan; p.MaxFrames = 1; return p }()},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if result, err := GenerateTwoFramesCPU(tc.plan, tc.talker, tc.predictor, tc.decoder); err == nil || result.Waveform != nil {
				t.Fatalf("accepted malformed two-frame request: err=%v", err)
			}
		})
	}
}

func TestTwoFrameCPUReleased(t *testing.T) {
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
		ModelRevision                   string   `json:"model_revision"`
		ModelSafetensorsSize            int64    `json:"model_safetensors_size"`
		ModelSafetensorsSHA256          string   `json:"model_safetensors_sha256"`
		ModelConfigSHA256               string   `json:"model_config_sha256"`
		SpeechTokenizerModelSize        int64    `json:"speech_tokenizer_model_size"`
		SpeechTokenizerModelSHA256      string   `json:"speech_tokenizer_model_sha256"`
		SpeechTokenizerConfigSHA256     string   `json:"speech_tokenizer_config_sha256"`
		TwoFrameOracleScriptSHA256      string   `json:"two_frame_oracle_script_sha256"`
		FirstSemanticToken              uint32   `json:"first_semantic_token"`
		SecondSemanticToken             uint32   `json:"second_semantic_token"`
		AcousticFrame                   []uint32 `json:"acoustic_frame"`
		SecondAcousticFrame             []uint32 `json:"second_acoustic_frame"`
		SecondHiddenSHA256              string   `json:"second_hidden_sha256"`
		SecondLogitsSHA256              string   `json:"second_logits_sha256"`
		SecondAcousticU32LESHA256       string   `json:"second_acoustic_u32le_sha256"`
		TwoFrameWaveformSHA256          string   `json:"two_frame_waveform_sha256"`
		SecondHiddenMaxAbsThreshold     float64  `json:"second_hidden_max_abs_threshold"`
		SecondLogitsMaxAbsThreshold     float64  `json:"second_logits_max_abs_threshold"`
		TwoFrameWaveformMaxAbsThreshold float64  `json:"two_frame_waveform_max_abs_threshold"`
	}
	if err := json.Unmarshal(data, &ref); err != nil {
		t.Fatal(err)
	}
	if ref.ModelRevision != "85e237c12c027371202489a0ec509ded67b5e4b5" || ref.ModelSafetensorsSize != 1811626576 || ref.ModelSafetensorsSHA256 != "bc3c7e785eb961179c25450d1acff03f839e0002f2f3a5aeb67b5735c0fa2adb" || ref.ModelConfigSHA256 != "81aca2b6fac304944d8acf345272d8a9a727d5fc2e2e66b222ab4729340c7455" || ref.SpeechTokenizerModelSize != 682293092 || ref.SpeechTokenizerModelSHA256 != "836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258" || ref.SpeechTokenizerConfigSHA256 != "ee65bb901c876664ab8707c487157aa1a6ee57c65969b28fb5ec9dc211e68167" || ref.TwoFrameOracleScriptSHA256 != "d31876390a74cc5135a323ea5ee3650da10c5547190312657e974fdfb7300e45" || ref.SecondHiddenSHA256 != "4f2f13934019f1c0db64e52d1a1e653fb38873d80db69a6f38b77f06633f2c97" || ref.SecondLogitsSHA256 != "465f7d75e54e0ad5f454d7dfa5c4d9336c5f4e10bab4fa55324974ce5cb14eda" || ref.SecondAcousticU32LESHA256 != "5075b441feca2dedddc07af3ec575837404a1f7747e996150c9559cd6367304e" || ref.TwoFrameWaveformSHA256 != "4671fa00c9511ded61ae396ae6cc74e5d15cf92062f6cc1f05a840ad3698cf46" || ref.SecondHiddenMaxAbsThreshold != 7e-5 || ref.SecondLogitsMaxAbsThreshold != 4e-5 || ref.TwoFrameWaveformMaxAbsThreshold != 1e-8 || ref.FirstSemanticToken != 1995 || ref.SecondSemanticToken != 215 {
		t.Fatal("unexpected pinned two-frame provenance")
	}
	for _, file := range []struct {
		path, sha string
		size      int64
	}{
		{filepath.Join(dir, "model.safetensors"), ref.ModelSafetensorsSHA256, ref.ModelSafetensorsSize},
		{filepath.Join(dir, "config.json"), ref.ModelConfigSHA256, 0},
		{filepath.Join(dir, "speech_tokenizer", "model.safetensors"), ref.SpeechTokenizerModelSHA256, ref.SpeechTokenizerModelSize},
		{filepath.Join(dir, "speech_tokenizer", "config.json"), ref.SpeechTokenizerConfigSHA256, 0},
	} {
		if err := verifyReleasedFile(file.path, file.sha, file.size); err != nil {
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
	prompt, err := BuildCustomVoicePrompt(tok, "Hello world", Ryan, English)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := NewRuntimeRequestPlan(cfg, RuntimeRequest{Conditioning: ConditioningRequest{Speaker: Ryan, Language: English}, Prompt: prompt, MaxFrames: 2})
	if err != nil {
		t.Fatal(err)
	}
	result, err := GenerateTwoFramesCPU(plan, talker, predictor, decoder)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(result.Semantic, []uint32{ref.FirstSemanticToken, ref.SecondSemanticToken}) || len(result.Acoustic) != 30 || !reflect.DeepEqual(result.Acoustic[:15], ref.AcousticFrame) || !reflect.DeepEqual(result.Acoustic[15:], ref.SecondAcousticFrame) {
		t.Fatalf("two-frame codes semantic=%v acoustic=%v", result.Semantic, result.Acoustic)
	}
	acousticBytes, err := os.ReadFile(filepath.Join(root, "second_acoustic.u32le"))
	if err != nil {
		t.Fatal(err)
	}
	if len(acousticBytes) != 60 || hashCP(acousticBytes) != ref.SecondAcousticU32LESHA256 {
		t.Fatal("invalid second acoustic reference")
	}
	for i, want := range result.Acoustic[15:] {
		if binary.LittleEndian.Uint32(acousticBytes[i*4:]) != want {
			t.Fatalf("acoustic %d mismatch", i)
		}
	}
	for _, row := range []struct {
		name   string
		values []float32
		sha    string
		limit  float64
	}{
		{"second_hidden", result.SecondHidden, ref.SecondHiddenSHA256, ref.SecondHiddenMaxAbsThreshold},
		{"second_logits", result.SecondLogits, ref.SecondLogitsSHA256, ref.SecondLogitsMaxAbsThreshold},
		{"two_frame_waveform", result.Waveform, ref.TwoFrameWaveformSHA256, ref.TwoFrameWaveformMaxAbsThreshold},
	} {
		blob, err := os.ReadFile(filepath.Join(root, row.name+".f32le"))
		if err != nil {
			t.Fatal(err)
		}
		if len(blob) != len(row.values)*4 || hashCP(blob) != row.sha {
			t.Fatalf("invalid %s oracle hash or length", row.name)
		}
		var maxAbs float64
		for i, x := range row.values {
			want := math.Float32frombits(binary.LittleEndian.Uint32(blob[i*4:]))
			if !finiteTalkerValue(x) || !finiteTalkerValue(want) {
				t.Fatalf("%s[%d] non-finite", row.name, i)
			}
			d := math.Abs(float64(x) - float64(want))
			if d > maxAbs {
				maxAbs = d
			}
		}
		if maxAbs > row.limit {
			t.Fatalf("%s max_abs=%g exceeds %g", row.name, maxAbs, row.limit)
		}
		t.Logf("%s count=%d max_abs=%g threshold=%g", row.name, len(row.values), maxAbs, row.limit)
	}
	if len(result.Waveform) != 3840 {
		t.Fatalf("two-frame samples=%d want=3840", len(result.Waveform))
	}
}
