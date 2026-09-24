package qwen3tts

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"sync"
	"testing"
)

func tinyPredictorSource(cfg ParsedConfig) fakeTalkerTensorSource {
	src := fakeTalkerTensorSource{}
	add := func(name string, dims []int, values []float32) {
		src[name] = struct {
			data  []float32
			shape []int
		}{values, dims}
	}
	const prefix = "talker.code_predictor."
	for i := 0; i < 15; i++ {
		emb := make([]float32, cfg.CPVocabSize*cfg.CPHiddenSize)
		emb[i*cfg.CPHiddenSize] = 1
		add(prefix+"model.codec_embedding."+strconv.Itoa(i)+".weight", []int{cfg.CPVocabSize, cfg.CPHiddenSize}, emb)
		head := make([]float32, cfg.CPVocabSize*cfg.CPHiddenSize)
		head[i*cfg.CPHiddenSize] = 1
		add(prefix+"lm_head."+strconv.Itoa(i)+".weight", []int{cfg.CPVocabSize, cfg.CPHiddenSize}, head)
	}
	add(prefix+"model.norm.weight", []int{cfg.CPHiddenSize}, []float32{1, 1, 1, 1})
	base := prefix + "model.layers.0"
	add(base+".input_layernorm.weight", []int{4}, []float32{1, 1, 1, 1})
	add(base+".post_attention_layernorm.weight", []int{4}, []float32{1, 1, 1, 1})
	for _, name := range []string{"q_proj", "k_proj", "v_proj", "o_proj"} {
		add(base+".self_attn."+name+".weight", []int{4, 4}, make([]float32, 16))
	}
	add(base+".self_attn.q_norm.weight", []int{4}, []float32{1, 1, 1, 1})
	add(base+".self_attn.k_norm.weight", []int{4}, []float32{1, 1, 1, 1})
	for _, name := range []string{"gate_proj", "up_proj", "down_proj"} {
		add(base+".mlp."+name+".weight", []int{4, 4}, make([]float32, 16))
	}
	return src
}

func TestCodePredictorCPUSynthetic(t *testing.T) {
	cfg := tinyTalkerConfig()
	src := tinyPredictorSource(cfg)
	predictor, err := LoadCodePredictorCPU(src, cfg)
	if err != nil {
		t.Fatal(err)
	}
	talker, err := LoadTalkerCPU(tinyTalkerSource(cfg), cfg)
	if err != nil {
		t.Fatal(err)
	}
	first, candidate, err := predictor.FirstAcousticFrame(talker, []float32{1, 2, 3, 4}, 7)
	if err != nil || len(first) != 15 || len(candidate) != cfg.CPVocabSize {
		t.Fatalf("codes=%v logits=%d err=%v", first, len(candidate), err)
	}
	// No global state: identical calls return owned, stable outputs.
	again, _, err := predictor.FirstAcousticFrame(talker, []float32{1, 2, 3, 4}, 7)
	if err != nil || !reflect.DeepEqual(first, again) {
		t.Fatalf("repeat=%v err=%v", again, err)
	}
	again[0]++
	if first[0] == again[0] {
		t.Fatal("aliased output")
	}
	for _, bad := range [][]float32{nil, {1, 2, 3}, {1, 2, 3, float32(math.Inf(1))}} {
		if out, _, err := predictor.FirstAcousticFrame(talker, bad, 7); err == nil || out != nil {
			t.Fatalf("accepted invalid hidden=%v", bad)
		}
	}
	if _, _, err := predictor.FirstAcousticFrame(talker, []float32{1, 2, 3, 4}, uint32(cfg.TalkerVocabSize)); err == nil {
		t.Fatal("accepted invalid semantic")
	}
	delete(src, "talker.code_predictor.model.norm.weight")
	if _, err := LoadCodePredictorCPU(src, cfg); err == nil {
		t.Fatal("accepted missing norm")
	}
	cfg.ModelSize = "1b7"
	if _, err := LoadCodePredictorCPU(tinyPredictorSource(cfg), cfg); err == nil {
		t.Fatal("accepted unimplemented 1.7B projection")
	}
}

func TestCodePredictorWorkspaceResetOwnershipAndConcurrency(t *testing.T) {
	cfg := tinyTalkerConfig()
	talker, err := LoadTalkerCPU(tinyTalkerSource(cfg), cfg)
	if err != nil {
		t.Fatal(err)
	}
	predictor, err := LoadCodePredictorCPU(tinyPredictorSource(cfg), cfg)
	if err != nil {
		t.Fatal(err)
	}
	w := newCodePredictorWorkspace(cfg, len(predictor.layers))
	if len(w.rope) == 0 {
		t.Fatal("missing RoPE")
	}
	inputs := [][]float32{{1, 2, 3, 4}, {-1, .5, 2, 3}, {2, 1, 0, -1}}
	wantCodes := make([][]uint32, len(inputs))
	wantLogits := make([][]float32, len(inputs))
	for i, input := range inputs {
		wantCodes[i], wantLogits[i], err = predictor.FirstAcousticFrame(talker, input, 7)
		if err != nil {
			t.Fatal(err)
		}
	}
	codes, logits, err := predictor.firstAcousticFrameWithWorkspace(talker, inputs[0], 7, w)
	if err != nil {
		t.Fatal(err)
	}
	ownedCodes, ownedLogits := append([]uint32(nil), codes...), append([]float32(nil), logits...)
	for _, i := range []int{1, 2, 0, 2, 1} {
		gotCodes, gotLogits, err := predictor.firstAcousticFrameWithWorkspace(talker, inputs[i], 7, w)
		if err != nil || !reflect.DeepEqual(gotCodes, wantCodes[i]) || !reflect.DeepEqual(gotLogits, wantLogits[i]) {
			t.Fatalf("frame %d after reset mismatch err=%v", i, err)
		}
		for layer := range w.kCache {
			if len(w.kCache[layer]) != 16*cfg.CPNumKeyValueHeads*cfg.CPHeadDim || len(w.vCache[layer]) != len(w.kCache[layer]) {
				t.Fatalf("frame %d layer %d KV length=%d/%d", i, layer, len(w.kCache[layer]), len(w.vCache[layer]))
			}
		}
	}
	if !reflect.DeepEqual(codes, ownedCodes) || !reflect.DeepEqual(logits, ownedLogits) {
		t.Fatal("prior outputs aliased workspace")
	}
	if _, _, err := predictor.firstAcousticFrameWithWorkspace(talker, []float32{1, 2}, 7, w); err == nil {
		t.Fatal("accepted malformed hidden")
	}
	gotCodes, gotLogits, err := predictor.firstAcousticFrameWithWorkspace(talker, inputs[0], 7, w)
	if err != nil || !reflect.DeepEqual(gotCodes, wantCodes[0]) || !reflect.DeepEqual(gotLogits, wantLogits[0]) {
		t.Fatalf("failed recovery after invalid input: %v", err)
	}
	const workers = 8
	results := make([][]uint32, workers)
	resultLogits := make([][]float32, workers)
	errs := make([]error, workers)
	var wg sync.WaitGroup
	for i := range results {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			results[i], resultLogits[i], errs[i] = predictor.FirstAcousticFrame(talker, inputs[i%len(inputs)], 7)
		}(i)
	}
	wg.Wait()
	for i := range results {
		if errs[i] != nil || !reflect.DeepEqual(results[i], wantCodes[i%len(inputs)]) || !reflect.DeepEqual(resultLogits[i], wantLogits[i%len(inputs)]) {
			t.Fatalf("concurrent frame %d err=%v", i, errs[i])
		}
	}
}

func TestCodePredictorReleasedFirstFrame(t *testing.T) {
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
		ModelRevision                      string   `json:"model_revision"`
		ModelSafetensorsSize               int64    `json:"model_safetensors_size"`
		ModelSafetensorsSHA256             string   `json:"model_safetensors_sha256"`
		ModelConfigSHA256                  string   `json:"model_config_sha256"`
		AcousticOracleScriptSHA256         string   `json:"acoustic_oracle_script_sha256"`
		AcousticFrame                      []uint32 `json:"acoustic_frame"`
		AcousticU32LESHA256                string   `json:"acoustic_u32le_sha256"`
		AcousticFirstLogitsShape           []int    `json:"acoustic_first_logits_shape"`
		AcousticFirstLogitsSHA256          string   `json:"acoustic_first_logits_sha256"`
		AcousticFirstLogitsMaxAbsThreshold float64  `json:"acoustic_first_logits_max_abs_threshold"`
	}
	if err := json.Unmarshal(data, &ref); err != nil {
		t.Fatal(err)
	}
	if ref.ModelRevision != "85e237c12c027371202489a0ec509ded67b5e4b5" || ref.ModelSafetensorsSHA256 != "bc3c7e785eb961179c25450d1acff03f839e0002f2f3a5aeb67b5735c0fa2adb" || ref.ModelSafetensorsSize != 1811626576 || ref.ModelConfigSHA256 != "81aca2b6fac304944d8acf345272d8a9a727d5fc2e2e66b222ab4729340c7455" || ref.AcousticOracleScriptSHA256 != "16cbbe0e5c9be29fc4fa831eeeb07cbbfef3b78d922b597eaff382b69b1b84f3" || ref.AcousticU32LESHA256 != "cb31dee74b8a5aa1ce79b240c423a37ef37b584d384a80e5556135aa354ca5a9" || ref.AcousticFirstLogitsSHA256 != "cb7b5634b924ef9f7df072d22c7f7a100935b51119bec1ca7933bc2a5c0ce3e4" || ref.AcousticFirstLogitsMaxAbsThreshold != 5e-5 {
		t.Fatal("unexpected acoustic reference provenance")
	}
	if err := verifyReleasedFile(filepath.Join(dir, "model.safetensors"), ref.ModelSafetensorsSHA256, ref.ModelSafetensorsSize); err != nil {
		t.Fatal(err)
	}
	if err := verifyReleasedFile(filepath.Join(dir, "config.json"), ref.ModelConfigSHA256, 0); err != nil {
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
	prompt := PromptIDs{Text: []uint32{151644, 77091, 198, 151671, 151671, 151671, 151671, 151671, 151672, 9707, 1879}, Codec: []uint32{2154, 2156, 2050, 2157, 3061, 2148, 2149}}
	plan, err := NewRuntimeRequestPlan(cfg, RuntimeRequest{Conditioning: ConditioningRequest{Speaker: Ryan, Language: English}, Prompt: prompt, MaxFrames: 1})
	if err != nil {
		t.Fatal(err)
	}
	talkerResult, err := talker.Prefill(plan)
	if err != nil {
		t.Fatal(err)
	}
	if talkerResult.SemanticToken != 1995 {
		t.Fatalf("talker semantic=%d", talkerResult.SemanticToken)
	}
	codes, logits, err := predictor.FirstAcousticFrame(talker, talkerResult.Hidden, talkerResult.SemanticToken)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(codes, ref.AcousticFrame) {
		t.Fatalf("acoustic=%v want=%v", codes, ref.AcousticFrame)
	}
	oracleCodes, err := os.ReadFile(filepath.Join(root, "acoustic.u32le"))
	if err != nil {
		t.Fatal(err)
	}
	if len(oracleCodes) != 60 || hashCP(oracleCodes) != ref.AcousticU32LESHA256 {
		t.Fatal("invalid pinned acoustic frame bytes")
	}
	for i, want := range codes {
		if binary.LittleEndian.Uint32(oracleCodes[4*i:]) != want {
			t.Fatalf("acoustic byte %d mismatch", i)
		}
	}
	oracle, err := os.ReadFile(filepath.Join(root, "acoustic_first_logits.f32le"))
	if err != nil {
		t.Fatal(err)
	}
	if len(ref.AcousticFirstLogitsShape) != 1 || len(logits) != ref.AcousticFirstLogitsShape[0] || len(oracle) != 4*len(logits) || hashCP(oracle) != ref.AcousticFirstLogitsSHA256 {
		t.Fatal("invalid pinned first acoustic logits")
	}
	var maxAbs float64
	for i, actual := range logits {
		want := math.Float32frombits(binary.LittleEndian.Uint32(oracle[4*i:]))
		if !finiteTalkerValue(actual) || !finiteTalkerValue(want) {
			t.Fatalf("nonfinite acoustic logit %d", i)
		}
		d := math.Abs(float64(actual) - float64(want))
		if d > maxAbs {
			maxAbs = d
		}
	}
	if maxAbs > ref.AcousticFirstLogitsMaxAbsThreshold {
		t.Fatalf("first acoustic logits max_abs=%g threshold=%g", maxAbs, ref.AcousticFirstLogitsMaxAbsThreshold)
	}
	t.Logf("first acoustic logits max_abs=%g threshold=%g", maxAbs, ref.AcousticFirstLogitsMaxAbsThreshold)
}

func hashCP(data []byte) string { h := sha256.Sum256(data); return fmt.Sprintf("%x", h) }
