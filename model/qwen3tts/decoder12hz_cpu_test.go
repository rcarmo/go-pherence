package qwen3tts

import (
	"fmt"
	"math"
	"strings"
	"testing"
)

type fakeDecoderSource map[string]struct {
	data  []float32
	shape []int
}

func (s fakeDecoderSource) GetFloat32(name string) ([]float32, []int, error) {
	tensor, ok := s[name]
	if !ok {
		return nil, nil, fmt.Errorf("missing %s", name)
	}
	return append([]float32(nil), tensor.data...), append([]int(nil), tensor.shape...), nil
}

func tinyDecoderSource(cfg Decoder12HzConfig) fakeDecoderSource {
	src := fakeDecoderSource{}
	add := func(name string, shape []int, values []float32) {
		src[name] = struct {
			data  []float32
			shape []int
		}{values, shape}
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
	for group := 0; group < cfg.Quantizers; group++ {
		prefix := "decoder.quantizer.rvq_first.vq.layers.0._codebook"
		if group > 0 {
			prefix = fmt.Sprintf("decoder.quantizer.rvq_rest.vq.layers.%d._codebook", group-1)
		}
		add(prefix+".embedding_sum", []int{cfg.CodebookSize, cfg.CodebookDim}, zeros(cfg.CodebookSize*cfg.CodebookDim))
		add(prefix+".cluster_usage", []int{cfg.CodebookSize}, ones(cfg.CodebookSize))
	}
	add("decoder.quantizer.rvq_first.output_proj.weight", []int{2 * cfg.CodebookDim, cfg.CodebookDim, 1}, zeros(2*cfg.CodebookDim*cfg.CodebookDim))
	add("decoder.quantizer.rvq_rest.output_proj.weight", []int{2 * cfg.CodebookDim, cfg.CodebookDim, 1}, zeros(2*cfg.CodebookDim*cfg.CodebookDim))
	addConv := func(prefix string, out, in, k int, weight []float32) {
		add(prefix+".weight", []int{out, in, k}, weight)
		add(prefix+".bias", []int{out}, zeros(out))
	}
	addLinear := func(prefix string, out, in int, bias bool) {
		add(prefix+".weight", []int{out, in}, identity(out, in))
		if bias {
			add(prefix+".bias", []int{out}, zeros(out))
		}
	}
	addConv("decoder.pre_conv.conv", cfg.LatentDim, 2*cfg.CodebookDim, 3, zeros(cfg.LatentDim*2*cfg.CodebookDim*3))
	addLinear("decoder.pre_transformer.input_proj", cfg.HiddenSize, cfg.LatentDim, true)
	addLinear("decoder.pre_transformer.output_proj", cfg.LatentDim, cfg.HiddenSize, true)
	for i := 0; i < cfg.Layers; i++ {
		p := fmt.Sprintf("decoder.pre_transformer.layers.%d", i)
		add(p+".input_layernorm.weight", []int{cfg.HiddenSize}, ones(cfg.HiddenSize))
		add(p+".post_attention_layernorm.weight", []int{cfg.HiddenSize}, ones(cfg.HiddenSize))
		add(p+".self_attn_layer_scale.scale", []int{cfg.HiddenSize}, zeros(cfg.HiddenSize))
		add(p+".mlp_layer_scale.scale", []int{cfg.HiddenSize}, zeros(cfg.HiddenSize))
		for _, n := range []string{"q_proj", "k_proj", "v_proj", "o_proj"} {
			addLinear(p+".self_attn."+n, cfg.HiddenSize, cfg.HiddenSize, false)
		}
		addLinear(p+".mlp.gate_proj", cfg.IntermediateSize, cfg.HiddenSize, false)
		addLinear(p+".mlp.up_proj", cfg.IntermediateSize, cfg.HiddenSize, false)
		addLinear(p+".mlp.down_proj", cfg.HiddenSize, cfg.IntermediateSize, false)
	}
	add("decoder.pre_transformer.norm.weight", []int{cfg.HiddenSize}, ones(cfg.HiddenSize))
	channels := cfg.LatentDim
	for i, rate := range cfg.PreUpsampleRates {
		p := fmt.Sprintf("decoder.upsample.%d", i)
		add(p+".0.conv.weight", []int{channels, channels, 2 * rate}, zeros(channels*channels*2*rate))
		add(p+".0.conv.bias", []int{channels}, zeros(channels))
		addConv(p+".1.dwconv.conv", channels, 1, 7, zeros(channels*7))
		add(p+".1.norm.weight", []int{channels}, ones(channels))
		add(p+".1.norm.bias", []int{channels}, zeros(channels))
		addLinear(p+".1.pwconv1", 4*channels, channels, true)
		addLinear(p+".1.pwconv2", channels, 4*channels, true)
		add(p+".1.gamma", []int{channels}, zeros(channels))
	}
	addConv("decoder.decoder.0.conv", cfg.DecoderDim, cfg.LatentDim, 7, zeros(cfg.DecoderDim*cfg.LatentDim*7))
	channels = cfg.DecoderDim
	addSnake := func(prefix string, ch int) {
		add(prefix+".alpha", []int{ch}, zeros(ch))
		add(prefix+".beta", []int{ch}, zeros(ch))
	}
	for i, rate := range cfg.DecoderUpsampleRates {
		out := channels / 2
		p := fmt.Sprintf("decoder.decoder.%d.block", i+1)
		addSnake(p+".0", channels)
		add(p+".1.conv.weight", []int{channels, out, 2 * rate}, zeros(channels*out*2*rate))
		add(p+".1.conv.bias", []int{out}, zeros(out))
		for u := 2; u <= 4; u++ {
			q := fmt.Sprintf("%s.%d", p, u)
			addSnake(q+".act1", out)
			addConv(q+".conv1.conv", out, out, 7, zeros(out*out*7))
			addSnake(q+".act2", out)
			addConv(q+".conv2.conv", out, out, 1, zeros(out*out))
		}
		channels = out
	}
	addSnake("decoder.decoder.5", channels)
	finalWeight := zeros(channels * 7)
	addConv("decoder.decoder.6.conv", 1, channels, 7, finalWeight)
	finalBias := src["decoder.decoder.6.conv.bias"]
	finalBias.data[0] = 2
	src["decoder.decoder.6.conv.bias"] = finalBias
	return src
}

func closeDecoderSlice(a, b []float32, tolerance float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Abs(float64(a[i]-b[i])) > tolerance {
			return false
		}
	}
	return true
}

func tinyDecoderPlan(t *testing.T, frames int) RuntimeRequestPlan {
	t.Helper()
	cfg, err := ParseConfig([]byte(`{"tts_model_type":"custom_voice","talker_config":{"hidden_size":1024,"num_attention_heads":16,"head_dim":64,"code_predictor_config":{"hidden_size":1024,"num_attention_heads":16,"head_dim":64,"vocab_size":2048,"num_code_groups":16}}}`))
	if err != nil {
		t.Fatal(err)
	}
	text, codec, err := CustomVoicePrefixIDs(123, Ryan, English)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := NewRuntimeRequestPlan(cfg, RuntimeRequest{Conditioning: ConditioningRequest{Speaker: Ryan, Language: English}, Prompt: PromptIDs{Text: text, Codec: codec}, MaxFrames: frames})
	if err != nil {
		t.Fatal(err)
	}
	return plan
}

func TestDecoder12HzCPUSyntheticShapeClampAndOwnership(t *testing.T) {
	cfg := tinyDecoder12HzConfig()
	src := tinyDecoderSource(cfg)
	model, err := LoadDecoder12HzCPU(src, cfg)
	if err != nil {
		t.Fatal(err)
	}
	// Mutating source storage after load must not change owned weights.
	tensor := src["decoder.decoder.6.conv.weight"]
	tensor.data[6] = 99
	src["decoder.decoder.6.conv.weight"] = tensor
	plan := tinyDecoderPlan(t, 1)
	semantic := []uint32{3071}
	acoustic := make([]uint32, 15)
	out, err := model.DecodeWaveform(plan, semantic, acoustic)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 1920 {
		t.Fatalf("samples=%d", len(out))
	}
	for i, v := range out {
		if v != 1 {
			t.Fatalf("sample[%d]=%g want clamped 1", i, v)
		}
	}
	again, err := model.DecodeWaveform(plan, semantic, acoustic)
	if err != nil || len(again) != len(out) {
		t.Fatalf("repeat err=%v len=%d", err, len(again))
	}
}

func TestDecoder12HzPrimitiveArithmetic(t *testing.T) {
	conv := decoderConv1D{weight: []float32{1, 2, 3}, bias: []float32{.5}, inChannels: 1, outChannels: 1, k: 3, dilation: 1}
	got, n, err := conv.forward([]float32{1, 2, 3}, 3)
	if err != nil || n != 3 || !closeDecoderSlice(got, []float32{3.5, 8.5, 14.5}, 1e-6) {
		t.Fatalf("causal conv=%v n=%d err=%v", got, n, err)
	}
	trans := decoderTransConv1D{weight: []float32{1, 2, 3, 4}, bias: []float32{.5}, inChannels: 1, outChannels: 1, k: 4, stride: 2}
	got, n, err = trans.forward([]float32{1, 2}, 2)
	if err != nil || n != 4 || !closeDecoderSlice(got, []float32{1.5, 2.5, 5.5, 8.5}, 1e-6) {
		t.Fatalf("transposed conv=%v n=%d err=%v", got, n, err)
	}
	snake := snakeBeta{alpha: []float32{0}, beta: []float32{0}}
	got = []float32{0, .5, -1}
	if err := snake.forwardInPlace(got, 3); err != nil {
		t.Fatal(err)
	}
	want := []float32{0, .5 + float32(math.Pow(math.Sin(.5), 2)), -1 + float32(math.Pow(math.Sin(-1), 2))}
	if !closeDecoderSlice(got, want, 1e-6) {
		t.Fatalf("SnakeBeta=%v want=%v", got, want)
	}
}

func TestNormalizedDecoderCodebook(t *testing.T) {
	src := fakeDecoderSource{
		"x.embedding_sum": {data: []float32{2, 4, 3, 6}, shape: []int{2, 2}},
		"x.cluster_usage": {data: []float32{2, 3}, shape: []int{2}},
	}
	got, err := loadNormalizedCodebook(src, "x", 2, 2)
	if err != nil || !closeDecoderSlice(got, []float32{1, 2, 1, 2}, 0) {
		t.Fatalf("codebook=%v err=%v", got, err)
	}
}

func TestDecoder12HzCPURejectsMalformed(t *testing.T) {
	cfg := tinyDecoder12HzConfig()
	if _, err := LoadDecoder12HzCPU(nil, cfg); err == nil {
		t.Fatal("accepted nil source")
	}
	bad := tinyDecoderSource(cfg)
	tensor := bad["decoder.pre_conv.conv.weight"]
	tensor.shape = []int{1}
	bad["decoder.pre_conv.conv.weight"] = tensor
	if _, err := LoadDecoder12HzCPU(bad, cfg); err == nil || !strings.Contains(err.Error(), "shape=") {
		t.Fatalf("shape err=%v", err)
	}
	bad = tinyDecoderSource(cfg)
	tensor = bad["decoder.pre_conv.conv.weight"]
	tensor.data[0] = float32(math.NaN())
	bad["decoder.pre_conv.conv.weight"] = tensor
	if _, err := LoadDecoder12HzCPU(bad, cfg); err == nil || !strings.Contains(err.Error(), "non-finite") {
		t.Fatalf("finite err=%v", err)
	}
	model, err := LoadDecoder12HzCPU(tinyDecoderSource(cfg), cfg)
	if err != nil {
		t.Fatal(err)
	}
	plan := tinyDecoderPlan(t, 1)
	if _, err := model.DecodeWaveform(plan, nil, nil); err == nil {
		t.Fatal("accepted empty input")
	}
	if _, err := model.DecodeWaveform(plan, []uint32{1}, make([]uint32, 14)); err == nil {
		t.Fatal("accepted short acoustic input")
	}
}
