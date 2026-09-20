package qwen3tts

import (
	"math"
	"testing"
)

func sizingConfig(t *testing.T) ParsedConfig {
	t.Helper()
	c, err := ParseConfig([]byte(`{}`))
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func TestLayoutsRejectMatchingWrappedSizes(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	for _, hidden := range []int{maxInt/2 + 1, maxInt/3 + 1} {
		f := newFFNLayout("forged", hidden, 4, 3)
		if err := f.Validate(); err == nil {
			t.Fatal("FFN accepted wrapped sizes", f)
		}
	}
	c := sizingConfig(t)
	c.TalkerTextVocabSize = maxInt/2 + 1
	c.TalkerTextHiddenSize = 4
	if _, err := NewEmbeddingLayout(c); err == nil {
		t.Fatal("embedding accepted wrapped product")
	}
	c = sizingConfig(t)
	c.TalkerTextVocabSize = maxInt - 3
	c.TalkerTextHiddenSize = 1
	if _, err := NewEmbeddingLayout(c); err == nil {
		t.Fatal("embedding accepted wrapped total")
	}
	c = sizingConfig(t)
	c.CPIntermediateSize = maxInt
	if _, err := NewCodePredictorFFNLayout(c); err == nil {
		t.Fatal("CP FFN accepted wrapped product")
	}
	c = sizingConfig(t)
	c.TalkerNumHiddenLayers = maxInt
	if _, err := NewTalkerFFNLayout(c); err == nil {
		t.Fatal("talker FFN accepted wrapped layer total")
	}
}

func TestAttentionAndTransformerSizingRejectOverflow(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	a := AttentionLayout{Name: "test", HiddenSize: 1, Layers: maxInt, Heads: 1, KVHeads: 1, HeadDim: 1, QueriesPerKV: 1, RoPETheta: 10000, RMSNormEps: 1e-5}
	if _, err := a.KVFloatsPerToken(); err == nil {
		t.Fatal("wrapped KV floats accepted")
	}
	p := TransformerPlan{HiddenSize: 1, IntermediateSize: 1, Layers: maxInt, Heads: 1, KVHeads: 1, HeadDim: 1, VocabSize: 1, KVFloatsPerToken: -2}
	if err := p.Validate("test"); err == nil {
		t.Fatal("matching wrapped KV count accepted")
	}
	if _, err := p.KVBytes(1, 4); err == nil {
		t.Fatal("invalid plan passed byte sizing")
	}
	a.Layers = 1
	p.Layers = 1
	p.KVFloatsPerToken = 2
	if _, err := a.KVBytes(maxInt, maxInt); err == nil {
		t.Fatal("KV bytes overflow accepted")
	}
	if _, err := p.KVBytes(maxInt, maxInt); err == nil {
		t.Fatal("plan KV bytes overflow accepted")
	}
	for _, seq := range []int{0, 3} {
		want := int64(seq * 2 * 4)
		if got, err := a.KVBytes(seq, 4); err != nil || got != want {
			t.Fatal(got, err)
		}
		if got, err := p.KVBytes(seq, 4); err != nil || got != want {
			t.Fatal(got, err)
		}
	}
	// Positive wrap (not only negative/zero) must not pass dimension equality.
	heads := maxInt/2 + 2
	hidden := heads * 4
	a.Heads = heads
	a.HeadDim = 4
	a.HiddenSize = hidden
	a.QueriesPerKV = heads
	if err := a.Validate(); err == nil {
		t.Fatal("wrapped head product accepted")
	}
}

func TestConfigAndAttentionRejectNonfiniteControls(t *testing.T) {
	for _, v := range []float64{math.NaN(), math.Inf(1), math.Inf(-1), 0, -1} {
		for field := 0; field < 4; field++ {
			c := sizingConfig(t)
			fields := []*float64{&c.TalkerRMSNormEps, &c.TalkerRoPETheta, &c.CPRMSNormEps, &c.CPRoPETheta}
			*fields[field] = v
			if c.Validate() == nil {
				t.Fatalf("accepted field%d=%g", field, v)
			}
		}
		a := AttentionLayout{Name: "test", HiddenSize: 1, Layers: 1, Heads: 1, KVHeads: 1, HeadDim: 1, QueriesPerKV: 1, RoPETheta: 10000, RMSNormEps: v}
		if a.Validate() == nil {
			t.Fatal("accepted layout epsilon", v)
		}
	}
	c := sizingConfig(t)
	c.TalkerNumAttentionHeads = int(^uint(0)>>1)/2 + 2
	c.TalkerHeadDim = 4
	c.TalkerHiddenSize = c.TalkerNumAttentionHeads * 4
	c.TalkerNumKeyValueHeads = 1
	if c.Validate() == nil {
		t.Fatal("wrapped config head dims accepted")
	}
}

func TestFrameAndReferenceSizingRejectOverflow(t *testing.T) {
	c := sizingConfig(t)
	d, err := NewDecoderInputLayout(c)
	if err != nil {
		t.Fatal(err)
	}
	w, err := NewWaveformLayout(DecoderPlan{FrameRateHz: 12})
	if err != nil {
		t.Fatal(err)
	}
	s := SpeakerEncoderLayout{Present: true, EmbeddingDim: 1, EmbeddingFloats: 1, SampleRateHz: 24000, ReferenceChannels: 1, SamplesPerSecond: 24000}
	maxInt := int(^uint(0) >> 1)
	for name, fn := range map[string]func(int) (int, error){"codes": d.CodesForFrames, "waveform": w.SamplesForFrames, "reference": s.ReferenceSamples} {
		if _, err := fn(maxInt); err == nil {
			t.Fatal(name, "accepted overflow")
		}
		if _, err := fn(-1); err == nil {
			t.Fatal(name, "accepted negative")
		}
		if n, err := fn(0); err != nil || n != 0 {
			t.Fatal(name, n, err)
		}
	}
	s.SampleRateHz = maxInt/2 + 1
	s.ReferenceChannels = 4
	s.SamplesPerSecond = 0
	if s.Validate() == nil {
		t.Fatal("matching wrapped samples accepted")
	}
}

func TestPromptSizingRejectsMatchingWrappedCounts(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	p := PrefillLayout{TextTokens: maxInt/2 + 1, CodecTokens: 1, FirstTextIndex: CustomVoiceFirstTextIndex, OverlayPosition: CustomVoiceFirstTextIndex, TalkerHiddenSize: 4, EmbeddingFloats: 0}
	if p.Validate() == nil {
		t.Fatal("wrapped prefill accepted")
	}
	p.TextTokens = 16
	p.TalkerHiddenSize = 1
	p.EmbeddingFloats = 16
	if _, err := p.EmbeddingBytes(maxInt); err == nil {
		t.Fatal("prefill byte overflow accepted")
	}
	l := TalkerInputLayout{TextHiddenSize: 1, TalkerHiddenSize: 4, TextTokens: maxInt/2 + 1, CodecTokens: 1, OverlayPosition: CustomVoiceFirstTextIndex, ProjectionFloats: 4, FusedInputFloats: 0}
	if l.Validate() == nil {
		t.Fatal("wrapped fused input accepted")
	}
	c := sizingConfig(t)
	p.EmbeddingFloats++
	if _, err := NewTalkerInputLayout(c, p); err == nil {
		t.Fatal("invalid prefill input accepted")
	}
}

func TestRequestSecondsRejectNonfiniteAndOverflow(t *testing.T) {
	c := sizingConfig(t)
	c.ModelType = CustomVoice
	text, codec, err := CustomVoicePrefixIDs(123, Ryan, English)
	if err != nil {
		t.Fatal(err)
	}
	req := RuntimeRequest{Conditioning: ConditioningRequest{Speaker: Ryan, Language: English}, Prompt: PromptIDs{Text: text, Codec: codec}}
	for _, seconds := range []float64{math.NaN(), math.Inf(1), -1, math.MaxFloat64, float64(int(^uint(0) >> 1))} {
		req.MaxSeconds = seconds
		if _, err := NewRuntimeRequestPlan(c, req); err == nil {
			t.Fatal("accepted seconds", seconds)
		}
	}
	req.MaxSeconds = .5
	if p, err := NewRuntimeRequestPlan(c, req); err != nil || p.MaxFrames != 6 || p.MaxSamples != 12000 {
		t.Fatal(p, err)
	}
}
