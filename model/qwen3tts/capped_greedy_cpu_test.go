package qwen3tts

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
)

func TestCappedGreedyCPURejectsMalformed(t *testing.T) {
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
		name string
		t    *TalkerCPU
		p    *CodePredictorCPU
		d    *Decoder12HzCPU
		plan RuntimeRequestPlan
	}{
		{"nil talker", nil, predictor, decoder, plan},
		{"nil predictor", talker, nil, decoder, plan},
		{"nil decoder", talker, predictor, nil, plan},
		{"bad plan", talker, predictor, decoder, func() RuntimeRequestPlan { p := plan; p.MaxFrames = 0; return p }()},
		{"over cap", talker, predictor, decoder, func() RuntimeRequestPlan { p := plan; p.MaxFrames = 33; return p }()},
		{"budget mismatch", talker, predictor, decoder, func() RuntimeRequestPlan { p := plan; p.MaxSamples--; return p }()},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := GenerateCappedGreedyCPU(tc.plan, tc.t, tc.p, tc.d)
			if err == nil || got.Waveform != nil {
				t.Fatalf("accepted malformed capped request: %v", err)
			}
		})
	}
}

func TestCappedGreedyCPUFirstEOSRejectsWithoutOutput(t *testing.T) {
	cfg := tinyTalkerConfig()
	src := tinyTalkerSource(cfg)
	// Force EOS to dominate the first prefill. The test supplies valid model
	// geometry but refuses a zero-frame request rather than decoding it.
	head := src["talker.codec_head.weight"]
	for i := 0; i < 4; i++ {
		head.data[int(CodecEOS)*4+i] = 1000
	}
	src["talker.codec_head.weight"] = head
	talker, err := LoadTalkerCPU(src, cfg)
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
	if got, err := GenerateCappedGreedyCPU(plan, talker, predictor, decoder); err == nil || !strings.Contains(err.Error(), "first semantic is EOS") || got.Waveform != nil {
		t.Fatalf("accepted first EOS or wrong failure: %+v %v", got, err)
	}
}

func TestCappedGreedyCPUStopAfterCompleteFrame(t *testing.T) {
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
	count := 0
	choose := func(logits []float32, eos uint32) (uint32, error) {
		count++
		if count == 2 {
			return eos, nil
		}
		return greedyTalkerToken(logits, eos)
	}
	result, err := generateCappedGreedyCPU(plan, talker, predictor, decoder, choose)
	if err != nil {
		t.Fatal(err)
	}
	if count != 2 || len(result.Semantic) != 1 || len(result.Acoustic) != 15 || len(result.Waveform) != 1920 || len(result.ContinuationHidden) != 0 {
		t.Fatalf("EOS emitted partial frame: count=%d result=%+v", count, result)
	}
	// The cap can exceed the number of completed frames without a decoder
	// budget mismatch, and the returned waveform must retain its own data.
	longPlan, err := NewRuntimeRequestPlan(cfg, RuntimeRequest{Conditioning: ConditioningRequest{Speaker: Ryan, Language: English}, Prompt: plan.Prompt, MaxFrames: 8})
	if err != nil {
		t.Fatal(err)
	}
	count = 0
	short, err := generateCappedGreedyCPU(longPlan, talker, predictor, decoder, choose)
	if err != nil || count != 2 || len(short.Semantic) != 1 || len(short.Waveform) != 1920 {
		t.Fatalf("early EOS with eight-frame cap: semantic=%v samples=%d calls=%d err=%v", short.Semantic, len(short.Waveform), count, err)
	}
	original := append([]float32(nil), short.Waveform...)
	if _, err := GenerateTwoFramesCPU(plan, talker, predictor, decoder); err != nil {
		t.Fatalf("fixed-frame no-EOS regression: %v", err)
	}
	if !reflect.DeepEqual(original, short.Waveform) {
		t.Fatal("returned waveform aliased later request")
	}
	bad := func([]float32, uint32) (uint32, error) { return 0, errors.New("selection rejected") }
	if got, err := generateCappedGreedyCPU(plan, talker, predictor, decoder, bad); err == nil || got.Waveform != nil {
		t.Fatalf("accepted selection error: %v", err)
	}
	onePlan, err := NewRuntimeRequestPlan(cfg, RuntimeRequest{Conditioning: ConditioningRequest{Speaker: Ryan, Language: English}, Prompt: plan.Prompt, MaxFrames: 1})
	if err != nil {
		t.Fatal(err)
	}
	count = 0
	one, err := generateCappedGreedyCPU(onePlan, talker, predictor, decoder, choose)
	if err != nil || count != 1 || len(one.Semantic) != 1 || len(one.Waveform) != 1920 {
		t.Fatalf("one-frame cap semantic=%v samples=%d calls=%d err=%v", one.Semantic, len(one.Waveform), count, err)
	}
}

func TestCappedGreedyMinTwoCPUBehavior(t *testing.T) {
	cfg := tinyTalkerConfig()
	src := tinyTalkerSource(cfg)
	head := src["talker.codec_head.weight"]
	for i := 0; i < 4; i++ {
		head.data[int(CodecEOS)*4+i] = 1000
	}
	src["talker.codec_head.weight"] = head
	talker, err := LoadTalkerCPU(src, cfg)
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
	base := tinyTalkerPlan(t, cfg)
	plan, err := NewRuntimeRequestPlan(cfg, RuntimeRequest{Conditioning: ConditioningRequest{Speaker: Ryan, Language: English}, Prompt: base.Prompt, MaxFrames: 8})
	if err != nil {
		t.Fatal(err)
	}
	if got, err := GenerateCappedGreedyCPU(plan, talker, predictor, decoder); err == nil || got.Waveform != nil {
		t.Fatalf("greedy accepted first EOS: %v", err)
	}
	result, err := GenerateCappedGreedyMinTwoCPU(plan, talker, predictor, decoder)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Semantic) != 2 || len(result.Acoustic) != 30 || len(result.Waveform) != 3840 || len(result.ContinuationHidden) != 1 || len(result.ContinuationLogits) != 1 {
		t.Fatalf("minimum two did not stop at EOS: semantic=%v acoustic=%d samples=%d", result.Semantic, len(result.Acoustic), len(result.Waveform))
	}
	for _, token := range result.Semantic {
		if token == CodecEOS {
			t.Fatal("emitted EOS as acoustic frame")
		}
	}
	one, err := NewRuntimeRequestPlan(cfg, RuntimeRequest{Conditioning: ConditioningRequest{Speaker: Ryan, Language: English}, Prompt: base.Prompt, MaxFrames: 1})
	if err != nil {
		t.Fatal(err)
	}
	capped, err := GenerateCappedGreedyMinTwoCPU(one, talker, predictor, decoder)
	if err != nil || len(capped.Semantic) != 1 || len(capped.Waveform) != 1920 {
		t.Fatalf("one-frame cap must not require two frames: semantic=%v err=%v", capped.Semantic, err)
	}
}

func TestCappedGreedyCPUConcurrentRequestsOwnScratch(t *testing.T) {
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
	want, err := GenerateCappedGreedyCPU(plan, talker, predictor, decoder)
	if err != nil {
		t.Fatal(err)
	}
	const workers = 8
	results := make([]BoundedCPUResult, workers)
	errors := make([]error, workers)
	var wg sync.WaitGroup
	for i := range results {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			results[i], errors[i] = GenerateCappedGreedyCPU(plan, talker, predictor, decoder)
		}(i)
	}
	wg.Wait()
	for i, result := range results {
		if errors[i] != nil || !reflect.DeepEqual(result, want) {
			t.Fatalf("request %d differs: err=%v", i, errors[i])
		}
	}
	if len(results[0].Waveform) == 0 {
		t.Fatal("missing waveform")
	}
	results[0].Waveform[0]++
	if results[0].Waveform[0] == results[1].Waveform[0] {
		t.Fatal("request waveforms alias")
	}
}

func TestCappedGreedyCPUReleasedEightFrames(t *testing.T)   { testCappedGreedyCPUReleasedFrames(t, 8) }
func TestCappedGreedyCPUReleasedSixteenFrames(t *testing.T) { testCappedGreedyCPUReleasedFrames(t, 16) }

func testCappedGreedyCPUReleasedFrames(t *testing.T, frames int) {
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
		ModelRevision                       string   `json:"model_revision"`
		ModelSafetensorsSize                int64    `json:"model_safetensors_size"`
		ModelSafetensorsSHA256              string   `json:"model_safetensors_sha256"`
		ModelConfigSHA256                   string   `json:"model_config_sha256"`
		SpeechTokenizerModelSize            int64    `json:"speech_tokenizer_model_size"`
		SpeechTokenizerModelSHA256          string   `json:"speech_tokenizer_model_sha256"`
		SpeechTokenizerConfigSHA256         string   `json:"speech_tokenizer_config_sha256"`
		EightFrameOracleScriptSHA256        string   `json:"eight_frame_oracle_script_sha256"`
		EightFrameSemantic                  []uint32 `json:"eight_frame_semantic"`
		EightFrameCodesSHA256               string   `json:"eight_frame_codes_sha256"`
		EightFrameWaveformSHA256            string   `json:"eight_frame_waveform_sha256"`
		EightFrameWaveformMaxAbsThreshold   float64  `json:"eight_frame_waveform_max_abs_threshold"`
		SixteenFrameOracleScriptSHA256      string   `json:"sixteen_frame_oracle_script_sha256"`
		SixteenFrameSemantic                []uint32 `json:"sixteen_frame_semantic"`
		SixteenFrameCodesSHA256             string   `json:"sixteen_frame_codes_sha256"`
		SixteenFrameWaveformSHA256          string   `json:"sixteen_frame_waveform_sha256"`
		SixteenFrameWaveformMaxAbsThreshold float64  `json:"sixteen_frame_waveform_max_abs_threshold"`
	}
	if err := json.Unmarshal(data, &ref); err != nil {
		t.Fatal(err)
	}
	if ref.ModelRevision != "85e237c12c027371202489a0ec509ded67b5e4b5" || ref.ModelSafetensorsSize != 1811626576 || ref.ModelSafetensorsSHA256 != "bc3c7e785eb961179c25450d1acff03f839e0002f2f3a5aeb67b5735c0fa2adb" || ref.ModelConfigSHA256 != "81aca2b6fac304944d8acf345272d8a9a727d5fc2e2e66b222ab4729340c7455" || ref.SpeechTokenizerModelSize != 682293092 || ref.SpeechTokenizerModelSHA256 != "836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258" || ref.SpeechTokenizerConfigSHA256 != "ee65bb901c876664ab8707c487157aa1a6ee57c65969b28fb5ec9dc211e68167" || ref.EightFrameOracleScriptSHA256 != "2d76facf4b4f3c1e4f4e9a81e637dc037a3aa08df10002c832a75c355cc60003" || ref.EightFrameCodesSHA256 != "fda03df846ac7a4d295e653850a18dfd3c26fbe3b768e476ba8ae3dd46bdaec3" || ref.EightFrameWaveformSHA256 != "03b796022c1a9f7a3ab874f1ce93d62c65e19dc7743df63e8ccc9ad55f341b12" || ref.EightFrameWaveformMaxAbsThreshold != 1e-6 {
		t.Fatal("unexpected pinned eight-frame provenance")
	}
	if frames != 8 && frames != 16 {
		t.Fatalf("unsupported released frame count %d", frames)
	}
	if frames == 16 && (ref.SixteenFrameOracleScriptSHA256 != "139a16f3915192df85ad55cef08be8b8ecff6978fda6b4b77b4bea7150f10cb3" || ref.SixteenFrameCodesSHA256 != "b68ff8305ab1bb3bd53cb251ab5cbe5f87c2a3772534b5d99ced9ea8bc423390" || ref.SixteenFrameWaveformSHA256 != "98aee9857bb6a73b690f2345765b00ed14469213fe1065821f8a13d2b1cfb973" || ref.SixteenFrameWaveformMaxAbsThreshold != 1.3e-6 || !reflect.DeepEqual(ref.SixteenFrameSemantic, []uint32{1995, 215, 212, 1181, 462, 251, 530, 122, 1792, 1792, 1086, 1086, 1086, 1724, 1724, 1792})) {
		t.Fatal("unexpected pinned sixteen-frame provenance")
	}
	prefix, script := "eight_frame", "qwen3tts_oracle_eight_frames.rs"
	semantic, codeSHA, waveformSHA, scriptSHA, threshold := ref.EightFrameSemantic, ref.EightFrameCodesSHA256, ref.EightFrameWaveformSHA256, ref.EightFrameOracleScriptSHA256, ref.EightFrameWaveformMaxAbsThreshold
	if frames == 16 {
		prefix, script = "sixteen_frame", "qwen3tts_oracle_sixteen_frames.rs"
		semantic, codeSHA, waveformSHA, scriptSHA, threshold = ref.SixteenFrameSemantic, ref.SixteenFrameCodesSHA256, ref.SixteenFrameWaveformSHA256, ref.SixteenFrameOracleScriptSHA256, ref.SixteenFrameWaveformMaxAbsThreshold
	}
	if err := verifyReleasedFile(filepath.Join("..", "..", "scripts", script), scriptSHA, 0); err != nil {
		t.Fatal(err)
	}
	for _, item := range []struct {
		path, sha string
		size      int64
	}{
		{filepath.Join(dir, "model.safetensors"), ref.ModelSafetensorsSHA256, ref.ModelSafetensorsSize},
		{filepath.Join(dir, "config.json"), ref.ModelConfigSHA256, 0},
		{filepath.Join(dir, "speech_tokenizer", "model.safetensors"), ref.SpeechTokenizerModelSHA256, ref.SpeechTokenizerModelSize},
		{filepath.Join(dir, "speech_tokenizer", "config.json"), ref.SpeechTokenizerConfigSHA256, 0},
	} {
		if err := verifyReleasedFile(item.path, item.sha, item.size); err != nil {
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
	plan, err := NewRuntimeRequestPlan(cfg, RuntimeRequest{Conditioning: ConditioningRequest{Speaker: Ryan, Language: English}, Prompt: prompt, MaxFrames: frames})
	if err != nil {
		t.Fatal(err)
	}
	result, err := GenerateCappedGreedyCPU(plan, talker, predictor, decoder)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(result.Semantic, semantic) || len(result.Acoustic) != frames*15 || len(result.Waveform) != frames*1920 {
		t.Fatalf("%d-frame geometry semantic=%v acoustic=%d samples=%d", frames, result.Semantic, len(result.Acoustic), len(result.Waveform))
	}
	codes, err := os.ReadFile(filepath.Join(root, prefix+"_codes.u32le"))
	if err != nil {
		t.Fatal(err)
	}
	if len(codes) != frames*16*4 || hashCP(codes) != codeSHA {
		t.Fatal("invalid reference codes")
	}
	for f := 0; f < frames; f++ {
		if want := binary.LittleEndian.Uint32(codes[f*64:]); want != result.Semantic[f] {
			t.Fatalf("semantic frame %d=%d want=%d", f, result.Semantic[f], want)
		}
		for j := 0; j < 15; j++ {
			want := binary.LittleEndian.Uint32(codes[f*64+(j+1)*4:])
			if result.Acoustic[f*15+j] != want {
				t.Fatalf("acoustic frame=%d group=%d got=%d want=%d", f, j, result.Acoustic[f*15+j], want)
			}
		}
	}
	wave, err := os.ReadFile(filepath.Join(root, prefix+"_waveform.f32le"))
	if err != nil {
		t.Fatal(err)
	}
	if len(wave) != len(result.Waveform)*4 || hashCP(wave) != waveformSHA {
		t.Fatal("invalid reference waveform")
	}
	var maxAbs float64
	for i, x := range result.Waveform {
		want := math.Float32frombits(binary.LittleEndian.Uint32(wave[i*4:]))
		if !finiteTalkerValue(x) || !finiteTalkerValue(want) {
			t.Fatalf("nonfinite sample %d", i)
		}
		d := math.Abs(float64(x) - float64(want))
		if d > maxAbs {
			maxAbs = d
		}
	}
	if maxAbs > threshold {
		t.Fatalf("waveform max_abs=%g threshold=%g", maxAbs, threshold)
	}
	t.Logf("%d-frame samples=%d max_abs=%g threshold=%g", frames, len(result.Waveform), maxAbs, threshold)
}
