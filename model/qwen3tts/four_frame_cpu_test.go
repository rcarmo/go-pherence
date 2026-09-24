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

func TestFourFrameCPURejectsMalformed(t *testing.T) {
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
	if got, err := GenerateFourFramesCPU(plan, talker, predictor, decoder); err == nil || got.Waveform != nil {
		t.Fatalf("accepted two-frame plan in four-frame entrypoint: %v", err)
	}
	plan.MaxFrames = 4
	if got, err := GenerateFourFramesCPU(plan, talker, predictor, decoder); err == nil || got.Waveform != nil {
		t.Fatalf("accepted inconsistent budgets: %v", err)
	}
}

func TestFourFrameCPUReleased(t *testing.T) {
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
		ModelRevision                    string   `json:"model_revision"`
		ModelSafetensorsSize             int64    `json:"model_safetensors_size"`
		ModelSafetensorsSHA256           string   `json:"model_safetensors_sha256"`
		ModelConfigSHA256                string   `json:"model_config_sha256"`
		SpeechTokenizerModelSize         int64    `json:"speech_tokenizer_model_size"`
		SpeechTokenizerModelSHA256       string   `json:"speech_tokenizer_model_sha256"`
		SpeechTokenizerConfigSHA256      string   `json:"speech_tokenizer_config_sha256"`
		FourFrameOracleScriptSHA256      string   `json:"four_frame_oracle_script_sha256"`
		FirstSemanticToken               uint32   `json:"first_semantic_token"`
		SecondSemanticToken              uint32   `json:"second_semantic_token"`
		ThirdSemanticToken               uint32   `json:"third_semantic_token"`
		FourthSemanticToken              uint32   `json:"fourth_semantic_token"`
		AcousticFrame                    []uint32 `json:"acoustic_frame"`
		SecondAcousticFrame              []uint32 `json:"second_acoustic_frame"`
		ThirdAcousticFrame               []uint32 `json:"third_acoustic_frame"`
		FourthAcousticFrame              []uint32 `json:"fourth_acoustic_frame"`
		FourthHiddenSHA256               string   `json:"fourth_hidden_sha256"`
		FourthLogitsSHA256               string   `json:"fourth_logits_sha256"`
		FourthAcousticU32LESHA256        string   `json:"fourth_acoustic_u32le_sha256"`
		FourFrameWaveformSHA256          string   `json:"four_frame_waveform_sha256"`
		FourthHiddenMaxAbsThreshold      float64  `json:"fourth_hidden_max_abs_threshold"`
		FourthLogitsMaxAbsThreshold      float64  `json:"fourth_logits_max_abs_threshold"`
		FourFrameWaveformMaxAbsThreshold float64  `json:"four_frame_waveform_max_abs_threshold"`
	}
	if err := json.Unmarshal(data, &ref); err != nil {
		t.Fatal(err)
	}
	if ref.ModelRevision != "85e237c12c027371202489a0ec509ded67b5e4b5" || ref.ModelSafetensorsSize != 1811626576 || ref.ModelSafetensorsSHA256 != "bc3c7e785eb961179c25450d1acff03f839e0002f2f3a5aeb67b5735c0fa2adb" || ref.ModelConfigSHA256 != "81aca2b6fac304944d8acf345272d8a9a727d5fc2e2e66b222ab4729340c7455" || ref.SpeechTokenizerModelSize != 682293092 || ref.SpeechTokenizerModelSHA256 != "836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258" || ref.SpeechTokenizerConfigSHA256 != "ee65bb901c876664ab8707c487157aa1a6ee57c65969b28fb5ec9dc211e68167" || ref.FourFrameOracleScriptSHA256 != "043107a4e7e53cc50e6cba03a22aa7654905cb479a09347b9632995492112134" || ref.FourthHiddenSHA256 != "d81ebc4210b48d6151b0c165a5903da36cf97d402b7a712865e906e0fcf78588" || ref.FourthLogitsSHA256 != "bad89b2e2947d8f313a5a4e10ecba7c73593e796b41a2fc5ffba8f9d008a7227" || ref.FourthAcousticU32LESHA256 != "d36877a9097255ee50b9eb28e1efac1aae092433d5ad3307c4671d05ae44ed9a" || ref.FourFrameWaveformSHA256 != "7b1ed8bcab0011dfa3d8fe90cfef9021fa6994cd4cdbf627bf87db840c48f304" || ref.FourthHiddenMaxAbsThreshold != 0.0001 || ref.FourthLogitsMaxAbsThreshold != 0.00004 || ref.FourFrameWaveformMaxAbsThreshold != 1e-8 || ref.FirstSemanticToken != 1995 || ref.SecondSemanticToken != 215 || ref.ThirdSemanticToken != 212 || ref.FourthSemanticToken != 1181 {
		t.Fatal("unexpected pinned four-frame provenance")
	}
	for _, asset := range []struct {
		path, sha string
		size      int64
	}{
		{filepath.Join(dir, "model.safetensors"), ref.ModelSafetensorsSHA256, ref.ModelSafetensorsSize},
		{filepath.Join(dir, "config.json"), ref.ModelConfigSHA256, 0},
		{filepath.Join(dir, "speech_tokenizer", "model.safetensors"), ref.SpeechTokenizerModelSHA256, ref.SpeechTokenizerModelSize},
		{filepath.Join(dir, "speech_tokenizer", "config.json"), ref.SpeechTokenizerConfigSHA256, 0},
	} {
		if err := verifyReleasedFile(asset.path, asset.sha, asset.size); err != nil {
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
	plan, err := NewRuntimeRequestPlan(cfg, RuntimeRequest{Conditioning: ConditioningRequest{Speaker: Ryan, Language: English}, Prompt: prompt, MaxFrames: 4})
	if err != nil {
		t.Fatal(err)
	}
	result, err := GenerateFourFramesCPU(plan, talker, predictor, decoder)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(result.Semantic, []uint32{ref.FirstSemanticToken, ref.SecondSemanticToken, ref.ThirdSemanticToken, ref.FourthSemanticToken}) || len(result.Acoustic) != 60 || len(result.ContinuationHidden) != 3 || len(result.ContinuationLogits) != 3 || !reflect.DeepEqual(result.Acoustic[:15], ref.AcousticFrame) || !reflect.DeepEqual(result.Acoustic[15:30], ref.SecondAcousticFrame) || !reflect.DeepEqual(result.Acoustic[30:45], ref.ThirdAcousticFrame) || !reflect.DeepEqual(result.Acoustic[45:], ref.FourthAcousticFrame) {
		t.Fatalf("four-frame codes semantic=%v acoustic=%v", result.Semantic, result.Acoustic)
	}
	acousticBytes, err := os.ReadFile(filepath.Join(root, "fourth_acoustic.u32le"))
	if err != nil {
		t.Fatal(err)
	}
	if len(acousticBytes) != 60 || hashCP(acousticBytes) != ref.FourthAcousticU32LESHA256 {
		t.Fatal("invalid fourth acoustic reference")
	}
	for i, want := range result.Acoustic[45:] {
		if binary.LittleEndian.Uint32(acousticBytes[i*4:]) != want {
			t.Fatalf("fourth acoustic %d mismatch", i)
		}
	}
	for _, row := range []struct {
		name   string
		values []float32
		sha    string
		limit  float64
	}{
		{"fourth_hidden", result.ContinuationHidden[2], ref.FourthHiddenSHA256, ref.FourthHiddenMaxAbsThreshold},
		{"fourth_logits", result.ContinuationLogits[2], ref.FourthLogitsSHA256, ref.FourthLogitsMaxAbsThreshold},
		{"four_frame_waveform", result.Waveform, ref.FourFrameWaveformSHA256, ref.FourFrameWaveformMaxAbsThreshold},
	} {
		blob, err := os.ReadFile(filepath.Join(root, row.name+".f32le"))
		if err != nil {
			t.Fatal(err)
		}
		if len(blob) != 4*len(row.values) || hashCP(blob) != row.sha {
			t.Fatalf("invalid %s oracle hash/length", row.name)
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
			t.Fatalf("%s max_abs=%g threshold=%g", row.name, maxAbs, row.limit)
		}
		t.Logf("%s count=%d max_abs=%g threshold=%g", row.name, len(row.values), maxAbs, row.limit)
	}
	if len(result.Waveform) != 7680 {
		t.Fatalf("four-frame samples=%d want=7680", len(result.Waveform))
	}
}
