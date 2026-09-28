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

func TestThreeFrameCPURejectsMalformed(t *testing.T) {
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
	if got, err := GenerateThreeFramesCPU(plan, talker, predictor, decoder); err == nil || got.Waveform != nil {
		t.Fatalf("accepted two-frame request in three-frame entrypoint: %v", err)
	}
	plan.MaxFrames = 3 // Must fail even when only this field changes: sample/code budgets disagree.
	if got, err := GenerateThreeFramesCPU(plan, talker, predictor, decoder); err == nil || got.Waveform != nil {
		t.Fatalf("accepted invalid plan: %v", err)
	}
}

func TestThreeFrameCPUReleased(t *testing.T) {
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
		ModelRevision                     string   `json:"model_revision"`
		ModelSafetensorsSize              int64    `json:"model_safetensors_size"`
		ModelSafetensorsSHA256            string   `json:"model_safetensors_sha256"`
		ModelConfigSHA256                 string   `json:"model_config_sha256"`
		SpeechTokenizerModelSize          int64    `json:"speech_tokenizer_model_size"`
		SpeechTokenizerModelSHA256        string   `json:"speech_tokenizer_model_sha256"`
		SpeechTokenizerConfigSHA256       string   `json:"speech_tokenizer_config_sha256"`
		ThreeFrameOracleScriptSHA256      string   `json:"three_frame_oracle_script_sha256"`
		FirstSemanticToken                uint32   `json:"first_semantic_token"`
		SecondSemanticToken               uint32   `json:"second_semantic_token"`
		ThirdSemanticToken                uint32   `json:"third_semantic_token"`
		AcousticFrame                     []uint32 `json:"acoustic_frame"`
		SecondAcousticFrame               []uint32 `json:"second_acoustic_frame"`
		ThirdAcousticFrame                []uint32 `json:"third_acoustic_frame"`
		ThirdHiddenSHA256                 string   `json:"third_hidden_sha256"`
		ThirdLogitsSHA256                 string   `json:"third_logits_sha256"`
		ThirdAcousticU32LESHA256          string   `json:"third_acoustic_u32le_sha256"`
		ThreeFrameWaveformSHA256          string   `json:"three_frame_waveform_sha256"`
		ThirdHiddenMaxAbsThreshold        float64  `json:"third_hidden_max_abs_threshold"`
		ThirdLogitsMaxAbsThreshold        float64  `json:"third_logits_max_abs_threshold"`
		ThreeFrameWaveformMaxAbsThreshold float64  `json:"three_frame_waveform_max_abs_threshold"`
	}
	if err := json.Unmarshal(data, &ref); err != nil {
		t.Fatal(err)
	}
	if ref.ModelRevision != "85e237c12c027371202489a0ec509ded67b5e4b5" || ref.ModelSafetensorsSize != 1811626576 || ref.ModelSafetensorsSHA256 != "bc3c7e785eb961179c25450d1acff03f839e0002f2f3a5aeb67b5735c0fa2adb" || ref.ModelConfigSHA256 != "81aca2b6fac304944d8acf345272d8a9a727d5fc2e2e66b222ab4729340c7455" || ref.SpeechTokenizerModelSize != 682293092 || ref.SpeechTokenizerModelSHA256 != "836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258" || ref.SpeechTokenizerConfigSHA256 != "ee65bb901c876664ab8707c487157aa1a6ee57c65969b28fb5ec9dc211e68167" || ref.ThreeFrameOracleScriptSHA256 != "5796647bff0d4a8284f76ef1b270a65591194e749d2aa6ae7c499c380515dd2e" || ref.ThirdHiddenSHA256 != "64adb44855cc7f4bfac8c6d1b2a24484c49eae577e08e917325c2878626e0cd1" || ref.ThirdLogitsSHA256 != "4853bdb514c152406fb392a7c265627aa4301e070040e4b90874cf1edd6ba2bb" || ref.ThirdAcousticU32LESHA256 != "0b018545334e9e5f3ec88141e8ae01b8a6f0c302431fa1e93645cde048630ebb" || ref.ThreeFrameWaveformSHA256 != "c0f4a245428a582caabcce624a0a6c1aad2d572b9acf10357abe741e83c773cd" || ref.ThirdHiddenMaxAbsThreshold != 0.00014 || ref.ThirdLogitsMaxAbsThreshold != 0.00005 || ref.ThreeFrameWaveformMaxAbsThreshold != 1e-8 || ref.FirstSemanticToken != 1995 || ref.SecondSemanticToken != 215 || ref.ThirdSemanticToken != 212 {
		t.Fatal("unexpected pinned three-frame provenance")
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
	plan, err := NewRuntimeRequestPlan(cfg, RuntimeRequest{Conditioning: ConditioningRequest{Speaker: Ryan, Language: English}, Prompt: prompt, MaxFrames: 3})
	if err != nil {
		t.Fatal(err)
	}
	result, err := GenerateThreeFramesCPU(plan, talker, predictor, decoder)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(result.Semantic, []uint32{ref.FirstSemanticToken, ref.SecondSemanticToken, ref.ThirdSemanticToken}) || len(result.Acoustic) != 45 || len(result.ContinuationHidden) != 2 || len(result.ContinuationLogits) != 2 || !reflect.DeepEqual(result.Acoustic[:15], ref.AcousticFrame) || !reflect.DeepEqual(result.Acoustic[15:30], ref.SecondAcousticFrame) || !reflect.DeepEqual(result.Acoustic[30:], ref.ThirdAcousticFrame) {
		t.Fatalf("three-frame codes semantic=%v acoustic=%v", result.Semantic, result.Acoustic)
	}
	acousticBytes, err := os.ReadFile(filepath.Join(root, "third_acoustic.u32le"))
	if err != nil {
		t.Fatal(err)
	}
	if len(acousticBytes) != 60 || hashCP(acousticBytes) != ref.ThirdAcousticU32LESHA256 {
		t.Fatal("invalid third acoustic reference")
	}
	for i, want := range result.Acoustic[30:] {
		if binary.LittleEndian.Uint32(acousticBytes[i*4:]) != want {
			t.Fatalf("third acoustic %d mismatch", i)
		}
	}
	for _, row := range []struct {
		name   string
		values []float32
		sha    string
		limit  float64
	}{
		{"third_hidden", result.ContinuationHidden[1], ref.ThirdHiddenSHA256, ref.ThirdHiddenMaxAbsThreshold},
		{"third_logits", result.ContinuationLogits[1], ref.ThirdLogitsSHA256, ref.ThirdLogitsMaxAbsThreshold},
		{"three_frame_waveform", result.Waveform, ref.ThreeFrameWaveformSHA256, ref.ThreeFrameWaveformMaxAbsThreshold},
	} {
		blob, err := os.ReadFile(filepath.Join(root, row.name+".f32le"))
		if err != nil {
			t.Fatal(err)
		}
		if len(blob) != len(row.values)*4 || hashCP(blob) != row.sha {
			t.Fatalf("invalid %s oracle hash/length", row.name)
		}
		var maxAbs float64
		for i, x := range row.values {
			want := math.Float32frombits(binary.LittleEndian.Uint32(blob[i*4:]))
			if !finiteTalkerValue(x) || !finiteTalkerValue(want) {
				t.Fatalf("%s[%d] non-finite", row.name, i)
			}
			delta := math.Abs(float64(x) - float64(want))
			if delta > maxAbs {
				maxAbs = delta
			}
		}
		if maxAbs > row.limit {
			t.Fatalf("%s max_abs=%g threshold=%g", row.name, maxAbs, row.limit)
		}
		t.Logf("%s count=%d max_abs=%g threshold=%g", row.name, len(row.values), maxAbs, row.limit)
	}
	if len(result.Waveform) != 5760 {
		t.Fatalf("three-frame samples=%d want=5760", len(result.Waveform))
	}
}
