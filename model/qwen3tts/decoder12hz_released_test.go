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

// TestDecoder12HzReleasedFirstFrame checks one pinned full-width F32 decode
// from a released semantic+acoustic frame. It is opt-in and hashes both
// checkpoints before loading any tensor payload. One 80ms frame is not a
// speech-quality or multi-frame-generation result.
func TestDecoder12HzReleasedFirstFrame(t *testing.T) {
	const envName = "GO_PHERENCE_QWEN3TTS_0B6_CUSTOMVOICE_DIR"
	dir := os.Getenv(envName)
	if dir == "" {
		t.Skipf("set %s to the pinned Qwen3-TTS 0.6B CustomVoice directory", envName)
	}
	root := filepath.Join("testdata", "customvoice_0b6_ryan_hello")
	data, err := os.ReadFile(filepath.Join(root, "reference.json"))
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		ModelRevision               string   `json:"model_revision"`
		ModelSafetensorsSize        int64    `json:"model_safetensors_size"`
		ModelSafetensorsSHA256      string   `json:"model_safetensors_sha256"`
		ModelConfigSHA256           string   `json:"model_config_sha256"`
		SpeechTokenizerModelSize    int64    `json:"speech_tokenizer_model_size"`
		SpeechTokenizerModelSHA256  string   `json:"speech_tokenizer_model_sha256"`
		SpeechTokenizerConfigSHA256 string   `json:"speech_tokenizer_config_sha256"`
		WaveformOracleScriptSHA256  string   `json:"waveform_oracle_script_sha256"`
		WaveformShape               []int    `json:"waveform_shape"`
		WaveformSampleRate          int      `json:"waveform_sample_rate"`
		WaveformF32LESHA256         string   `json:"waveform_f32le_sha256"`
		WaveformMaxAbsThreshold     float64  `json:"waveform_max_abs_threshold"`
		AcousticFrame               []uint32 `json:"acoustic_frame"`
		FirstSemanticToken          uint32   `json:"first_semantic_token"`
	}
	if err := json.Unmarshal(data, &ref); err != nil {
		t.Fatal(err)
	}
	if ref.ModelRevision != "85e237c12c027371202489a0ec509ded67b5e4b5" || ref.ModelSafetensorsSize != 1811626576 || ref.ModelSafetensorsSHA256 != "bc3c7e785eb961179c25450d1acff03f839e0002f2f3a5aeb67b5735c0fa2adb" || ref.ModelConfigSHA256 != "81aca2b6fac304944d8acf345272d8a9a727d5fc2e2e66b222ab4729340c7455" || ref.SpeechTokenizerModelSize != 682293092 || ref.SpeechTokenizerModelSHA256 != "836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258" || ref.SpeechTokenizerConfigSHA256 != "ee65bb901c876664ab8707c487157aa1a6ee57c65969b28fb5ec9dc211e68167" || ref.WaveformOracleScriptSHA256 != "f8114ee628b4a9795c5c66ea99c0a3b1a0f3b2d1c6cb53038846bb9d1e2fcc6a" || ref.WaveformF32LESHA256 != "aaf863d69f21155ffe6a21b3b766b19424147ab9b7e8746e6ebd1a8f527957b0" || ref.WaveformMaxAbsThreshold != 1e-8 || ref.WaveformSampleRate != 24000 || ref.FirstSemanticToken != 1995 {
		t.Fatal("unexpected released waveform provenance")
	}
	for _, file := range []struct {
		path, sha string
		bytes     int64
	}{
		{filepath.Join(dir, "model.safetensors"), ref.ModelSafetensorsSHA256, ref.ModelSafetensorsSize},
		{filepath.Join(dir, "config.json"), ref.ModelConfigSHA256, 0},
		{filepath.Join(dir, "speech_tokenizer", "model.safetensors"), ref.SpeechTokenizerModelSHA256, ref.SpeechTokenizerModelSize},
		{filepath.Join(dir, "speech_tokenizer", "config.json"), ref.SpeechTokenizerConfigSHA256, 0},
	} {
		if err := verifyReleasedFile(file.path, file.sha, file.bytes); err != nil {
			t.Fatal(err)
		}
	}
	cfg, err := ReadModelDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	decoder, err := LoadDecoder12HzCPUFromDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	prompt := PromptIDs{Text: []uint32{151644, 77091, 198, 151671, 151671, 151671, 151671, 151671, 151672, 9707, 1879}, Codec: []uint32{2154, 2156, 2050, 2157, 3061, 2148, 2149}}
	plan, err := NewRuntimeRequestPlan(cfg, RuntimeRequest{Conditioning: ConditioningRequest{Speaker: Ryan, Language: English}, Prompt: prompt, MaxFrames: 1})
	if err != nil {
		t.Fatal(err)
	}
	if len(ref.AcousticFrame) != 15 {
		t.Fatal("missing released acoustic frame")
	}
	wave, err := decoder.DecodeWaveform(plan, []uint32{ref.FirstSemanticToken}, ref.AcousticFrame)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(ref.WaveformShape, []int{1920}) || len(wave) != ref.WaveformShape[0] {
		t.Fatalf("waveform length=%d want=%v", len(wave), ref.WaveformShape)
	}
	oracle, err := os.ReadFile(filepath.Join(root, "waveform.f32le"))
	if err != nil {
		t.Fatal(err)
	}
	if len(oracle) != len(wave)*4 || hashCP(oracle) != ref.WaveformF32LESHA256 {
		t.Fatal("invalid pinned waveform bytes")
	}
	var maxAbs float64
	for i, x := range wave {
		want := math.Float32frombits(binary.LittleEndian.Uint32(oracle[i*4:]))
		if !finiteTalkerValue(x) || !finiteTalkerValue(want) {
			t.Fatalf("nonfinite sample %d", i)
		}
		delta := math.Abs(float64(x) - float64(want))
		if delta > maxAbs {
			maxAbs = delta
		}
	}
	if maxAbs > ref.WaveformMaxAbsThreshold {
		t.Fatalf("waveform max_abs=%g threshold=%g", maxAbs, ref.WaveformMaxAbsThreshold)
	}
	t.Logf("waveform samples=%d max_abs=%g threshold=%g", len(wave), maxAbs, ref.WaveformMaxAbsThreshold)
}
