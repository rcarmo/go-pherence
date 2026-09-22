package minicpmv

import (
	"fmt"
	"math"
	"strings"
	"testing"

	audioload "github.com/rcarmo/go-pherence/loader/audio"
	"github.com/rcarmo/go-pherence/loader/config"
)

func tinyAudioConfig() config.MiniCPMVConfig {
	return config.MiniCPMVConfig{
		Architectures: []string{"MiniCPMO"}, ModelType: "minicpmo",
		HiddenSize: 4, NumHiddenLayers: 1, NumAttentionHeads: 1, VocabSize: 8, QueryNum: 1,
		AudioPoolStep: 2,
		AudioConfig: &config.MiniCPMOAudioConfig{
			ModelType: "whisper", DModel: 4, EncoderLayers: 1, EncoderHeads: 1,
			EncoderFFNDim: 16, FeatureSize: 80, NumMelBins: 80, SamplingRate: 16000,
			MaxSourcePositions: 1500, ActivationFunction: "gelu",
		},
	}
}

func tinyAudioSource(cfg config.MiniCPMVConfig) fakeTextTensorSource {
	s := cfg.MiniCPMVSummary()
	src := fakeTextTensorSource{}
	add := func(name string, shape []int, data []float32) {
		src[name] = struct {
			data  []float32
			shape []int
		}{data: data, shape: shape}
	}
	zeros := func(n int) []float32 { return make([]float32, n) }
	ones := func(n int) []float32 {
		out := make([]float32, n)
		for i := range out {
			out[i] = 1
		}
		return out
	}
	identity := func(rows, cols int) []float32 {
		out := make([]float32, rows*cols)
		for i := 0; i < min(rows, cols); i++ {
			out[i*cols+i] = 1
		}
		return out
	}
	d, f, positions := s.AudioHiddenSize, s.AudioIntermediateSize, s.AudioMaxSourcePositions
	add("apm.conv1.weight", []int{d, s.AudioMelBins, 3}, zeros(d*s.AudioMelBins*3))
	add("apm.conv1.bias", []int{d}, zeros(d))
	add("apm.conv2.weight", []int{d, d, 3}, zeros(d*d*3))
	add("apm.conv2.bias", []int{d}, zeros(d))
	position := make([]float32, positions*d)
	for row := 0; row < positions; row++ {
		copy(position[row*d:(row+1)*d], []float32{1, 2, 3, 4})
	}
	add("apm.embed_positions.weight", []int{positions, d}, position)
	add("apm.layer_norm.weight", []int{d}, ones(d))
	add("apm.layer_norm.bias", []int{d}, zeros(d))
	for layer := 0; layer < s.AudioLayers; layer++ {
		prefix := fmt.Sprintf("apm.layers.%d.", layer)
		add(prefix+"self_attn_layer_norm.weight", []int{d}, ones(d))
		add(prefix+"self_attn_layer_norm.bias", []int{d}, zeros(d))
		add(prefix+"self_attn.q_proj.weight", []int{d, d}, zeros(d*d))
		add(prefix+"self_attn.q_proj.bias", []int{d}, zeros(d))
		add(prefix+"self_attn.k_proj.weight", []int{d, d}, zeros(d*d))
		add(prefix+"self_attn.v_proj.weight", []int{d, d}, zeros(d*d))
		add(prefix+"self_attn.v_proj.bias", []int{d}, zeros(d))
		add(prefix+"self_attn.out_proj.weight", []int{d, d}, zeros(d*d))
		add(prefix+"self_attn.out_proj.bias", []int{d}, zeros(d))
		add(prefix+"final_layer_norm.weight", []int{d}, ones(d))
		add(prefix+"final_layer_norm.bias", []int{d}, zeros(d))
		add(prefix+"fc1.weight", []int{f, d}, zeros(f*d))
		add(prefix+"fc1.bias", []int{f}, zeros(f))
		add(prefix+"fc2.weight", []int{d, f}, zeros(d*f))
		add(prefix+"fc2.bias", []int{d}, zeros(d))
	}
	add("audio_projection_layer.linear1.weight", []int{s.HiddenSize, d}, identity(s.HiddenSize, d))
	add("audio_projection_layer.linear1.bias", []int{s.HiddenSize}, zeros(s.HiddenSize))
	add("audio_projection_layer.linear2.weight", []int{s.HiddenSize, s.HiddenSize}, identity(s.HiddenSize, s.HiddenSize))
	add("audio_projection_layer.linear2.bias", []int{s.HiddenSize}, zeros(s.HiddenSize))
	return src
}

func TestAudioCPUSyntheticEncodePoolAndInject(t *testing.T) {
	cfg := tinyAudioConfig()
	model, err := LoadAudioCPU(tinyAudioSource(cfg), cfg)
	if err != nil {
		t.Fatal(err)
	}
	features := make([]float32, 80*8)
	before := append([]float32(nil), features...)
	out, err := model.EncodeAudio(features, 8, 80)
	if err != nil {
		t.Fatal(err)
	}
	// Conv2 gives ceil(8/2)=4 rows and AvgPool1d(k=2,s=2) gives two tokens.
	if len(out) != 2*4 {
		t.Fatalf("output values=%d want 8", len(out))
	}
	want := []float32{0, 0, 0.4472118, 1.3416355, 0, 0, 0.4472118, 1.3416355}
	if !closeTextSlice(out, want, 1e-5) {
		t.Fatalf("unexpected synthetic output: %v want %v", out, want)
	}
	if !closeTextSlice(features, before, 0) {
		t.Fatal("features mutated")
	}
	again, err := model.EncodeAudio(features, 8, 80)
	if err != nil || !closeTextSlice(out, again, 0) {
		t.Fatalf("nondeterministic output err=%v got=%v want=%v", err, again, out)
	}

	plan := AudioPromptPlan{PatchTokens: 2, AudioSpans: []AudioSpan{{PatchStart: 1, PatchEnd: 3}}}
	tokens := []float32{9, 9, 9, 9, 1, 1, 1, 1, 2, 2, 2, 2, 8, 8, 8, 8}
	injected, meta, err := model.EncodeAndInject(features, 8, 80, tokens, 4, 4, plan)
	if err != nil {
		t.Fatal(err)
	}
	if meta.ReplacedTokens != 2 || injected[0] != 9 || injected[12] != 8 {
		t.Fatalf("bad injection meta=%+v output=%v", meta, injected)
	}
	if tokens[4] != 1 {
		t.Fatal("token embeddings mutated")
	}
}

func TestAudioCPUFrontendAndInputValidation(t *testing.T) {
	cfg := tinyAudioConfig()
	model, err := LoadAudioCPU(tinyAudioSource(cfg), cfg)
	if err != nil {
		t.Fatal(err)
	}
	samples := make([]float32, 320)
	samples[5] = .5
	features, frames, err := model.Features(samples, 16000)
	if err != nil || frames != 2 || len(features) != 160 {
		t.Fatalf("frontend shape frames=%d values=%d err=%v", frames, len(features), err)
	}
	padded := make([]float32, 30*16000)
	copy(padded, samples)
	full, fullFrames, err := audioload.WhisperLogMel(padded, 80)
	if err != nil || fullFrames != 3000 {
		t.Fatalf("full frontend err=%v frames=%d", err, fullFrames)
	}
	for mel := 0; mel < 80; mel++ {
		for frame := 0; frame < frames; frame++ {
			if features[mel*frames+frame] != full[mel*fullFrames+frame] {
				t.Fatalf("trimmed frontend mismatch mel=%d frame=%d", mel, frame)
			}
		}
	}
	if _, _, err := model.Features(samples, 8000); err == nil {
		t.Fatal("accepted wrong sample rate")
	}
	badPCM := append([]float32(nil), samples...)
	badPCM[0] = float32(math.NaN())
	if _, _, err := model.Features(badPCM, 16000); err == nil || !strings.Contains(err.Error(), "non-finite") {
		t.Fatalf("bad PCM error=%v", err)
	}
	if _, err := model.EncodeAudio(make([]float32, 79*8), 8, 79); err == nil {
		t.Fatal("accepted wrong feature size")
	}
	badFeatures := make([]float32, 80*8)
	badFeatures[0] = float32(math.Inf(1))
	if _, err := model.EncodeAudio(badFeatures, 8, 80); err == nil || !strings.Contains(err.Error(), "non-finite") {
		t.Fatalf("bad feature error=%v", err)
	}
}

func TestLoadAudioCPURejectsPolicyShapeAndNonFiniteWeights(t *testing.T) {
	cfg := tinyAudioConfig()
	if _, err := LoadAudioCPU(nil, cfg); err == nil {
		t.Fatal("accepted nil source")
	}
	badPolicy := cfg
	copyAudio := *cfg.AudioConfig
	badPolicy.AudioConfig = &copyAudio
	badPolicy.AudioConfig.AttentionDropout = .1
	if _, err := LoadAudioCPU(tinyAudioSource(cfg), badPolicy); err == nil || !strings.Contains(err.Error(), "policy") {
		t.Fatalf("policy error=%v", err)
	}
	badEpsilon := cfg
	copyAudio = *cfg.AudioConfig
	badEpsilon.AudioConfig = &copyAudio
	badEpsilon.AudioConfig.LayerNormEps = 1e-6
	if _, err := LoadAudioCPU(tinyAudioSource(cfg), badEpsilon); err == nil || !strings.Contains(err.Error(), "layer_norm_eps") {
		t.Fatalf("epsilon error=%v", err)
	}
	badShape := tinyAudioSource(cfg)
	tensor := badShape["audio_projection_layer.linear1.weight"]
	tensor.shape = []int{3, 4}
	badShape["audio_projection_layer.linear1.weight"] = tensor
	if _, err := LoadAudioCPU(badShape, cfg); err == nil || !strings.Contains(err.Error(), "shape=") {
		t.Fatalf("shape error=%v", err)
	}
	badWeight := tinyAudioSource(cfg)
	tensor = badWeight["apm.conv1.weight"]
	tensor.data[0] = float32(math.NaN())
	badWeight["apm.conv1.weight"] = tensor
	if _, err := LoadAudioCPU(badWeight, cfg); err == nil || !strings.Contains(err.Error(), "non-finite") {
		t.Fatalf("weight error=%v", err)
	}
}

func TestCheckedAudioTensorSourceOwnsValues(t *testing.T) {
	cfg := tinyAudioConfig()
	src := tinyAudioSource(cfg)
	model, err := LoadAudioCPU(src, cfg)
	if err != nil {
		t.Fatal(err)
	}
	position := src["apm.embed_positions.weight"]
	position.data[0] = 999
	src["apm.embed_positions.weight"] = position
	projector := src["audio_projection_layer.linear1.weight"]
	projector.data[0] = 999
	src["audio_projection_layer.linear1.weight"] = projector
	out, err := model.EncodeAudio(make([]float32, 80*8), 8, 80)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{0, 0, 0.4472118, 1.3416355, 0, 0, 0.4472118, 1.3416355}
	if !closeTextSlice(out, want, 1e-5) {
		t.Fatalf("loaded weights alias source: %v", out)
	}
}

func TestAudioCPURejectsPromptMismatch(t *testing.T) {
	cfg := tinyAudioConfig()
	model, err := LoadAudioCPU(tinyAudioSource(cfg), cfg)
	if err != nil {
		t.Fatal(err)
	}
	features := make([]float32, 80*8)
	if _, _, err := model.EncodeAndInject(features, 8, 80, make([]float32, 16), 4, 4, AudioPromptPlan{}); err == nil {
		t.Fatal("accepted missing audio span")
	}
	plan := AudioPromptPlan{PatchTokens: 1, AudioSpans: []AudioSpan{{PatchStart: 1, PatchEnd: 2}}}
	if _, _, err := model.EncodeAndInject(features, 8, 80, make([]float32, 16), 4, 4, plan); err == nil || !strings.Contains(err.Error(), "mismatch") {
		t.Fatalf("prompt mismatch error=%v", err)
	}
}
