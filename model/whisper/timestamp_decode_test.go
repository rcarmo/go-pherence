package whisper

import "testing"

func TestSuppressTimestampPromptSpecialsAllowsTimestamps(t *testing.T) {
	logits := make([]float32, TokenTimestampBegin+3)
	for i := range logits {
		logits[i] = 1
	}
	suppressTimestampPromptSpecials(logits)
	if logits[TokenSOT] > -1e20 || logits[TokenTranslate] > -1e20 || logits[TokenNoTimestamps] > -1e20 {
		t.Fatalf("prompt/control tokens not suppressed")
	}
	if logits[TokenTimestampBegin] != 1 || logits[TokenTimestampBegin+2] != 1 {
		t.Fatalf("timestamp tokens were suppressed")
	}
}

func TestSuppressInvalidTimestampTransitionsForcesTimestampPair(t *testing.T) {
	logits := make([]float32, TokenTimestampBegin+8)
	for i := range logits {
		logits[i] = 1
	}
	suppressInvalidTimestampTransitions(logits, []int{42, TokenTimestampBegin + 5})
	if logits[42] > -1e20 || logits[TokenEOT] > -1e20 {
		t.Fatalf("text/EOT logits were not suppressed after unpaired timestamp")
	}
	if logits[TokenTimestampBegin+4] > -1e20 {
		t.Fatalf("decreasing timestamp was not suppressed")
	}
	if logits[TokenTimestampBegin+5] < 0 || logits[TokenTimestampBegin+7] < 0 {
		t.Fatalf("valid timestamp pair candidates were suppressed")
	}
}

func TestSuppressInvalidTimestampTransitionsStopsTimestampRunAfterPair(t *testing.T) {
	logits := make([]float32, TokenTimestampBegin+8)
	for i := range logits {
		logits[i] = 1
	}
	suppressInvalidTimestampTransitions(logits, []int{TokenTimestampBegin + 3, TokenTimestampBegin + 5})
	if logits[42] != 1 || logits[TokenEOT] != 1 {
		t.Fatalf("text/EOT logits should remain available after timestamp pair")
	}
	if logits[TokenTimestampBegin+5] > -1e20 || logits[TokenTimestampBegin+7] > -1e20 {
		t.Fatalf("timestamp logits were not suppressed after completed pair")
	}
}

func TestGreedyDecodeWithTimestampsPromptAcceptsTranslatePrompt(t *testing.T) {
	cfg := Tiny()
	cfg.DecoderLayers = 0
	cfg.MaxDecoderLength = 2
	dec := newZeroDecoderForTest(cfg)
	state := NewDecoderState(cfg, make([]float32, cfg.EncoderDModel), 1, dec)
	segments := GreedyDecodeWithTimestampsPrompt(dec, state, cfg, LanguageTokens["en"], TokenTranslate)
	if len(segments) == 0 {
		t.Fatal("expected timestamp prompt decode to flush at least one segment")
	}
}

func TestGreedyDecodeWithTimestampsStopsRepeatedTextRun(t *testing.T) {
	cfg := Tiny()
	cfg.DecoderLayers = 0
	cfg.MaxDecoderLength = 16
	dec := newZeroDecoderForTest(cfg)
	state := NewDecoderState(cfg, make([]float32, cfg.EncoderDModel), 1, dec)
	segments := GreedyDecodeWithTimestampsPrompt(dec, state, cfg, LanguageTokens["en"], TokenTranslate)
	if len(segments) != 1 {
		t.Fatalf("segments=%d want 1", len(segments))
	}
	if got := len(segments[0].Tokens); got >= cfg.MaxDecoderLength {
		t.Fatalf("timestamp text repeated until token limit: emitted %d tokens", got)
	}
}

func TestGreedyDecodeWithTimestampsFlushesAtTokenLimit(t *testing.T) {
	cfg := Tiny()
	cfg.MaxDecoderLength = 3
	dec := newZeroDecoderForTest(cfg)
	encLen := 2
	state := NewDecoderState(cfg, make([]float32, encLen*cfg.EncoderDModel), encLen, dec)

	segments := GreedyDecodeWithTimestamps(dec, state, cfg)
	if len(segments) != 1 {
		t.Fatalf("segments=%d want 1", len(segments))
	}
	if len(segments[0].Tokens) == 0 {
		t.Fatalf("expected flushed text tokens")
	}
	if segments[0].End <= segments[0].Start {
		t.Fatalf("invalid segment timing: %+v", segments[0])
	}
}

func newZeroDecoderForTest(cfg Config) *Decoder {
	dec := NewDecoder(cfg)
	dModel := cfg.DecoderDModel
	dec.TokenEmbed = make([]float32, cfg.VocabSize*dModel)
	dec.PosEmbed = make([]float32, cfg.MaxDecoderLength*dModel)
	dec.FinalLNWeight = ones(dModel)
	dec.FinalLNBias = make([]float32, dModel)
	for i := range dec.Layers {
		l := &dec.Layers[i]
		l.SelfAttnLNWeight = ones(dModel)
		l.SelfAttnLNBias = make([]float32, dModel)
		l.SelfQWeight = make([]float32, dModel*dModel)
		l.SelfQBias = make([]float32, dModel)
		l.SelfKWeight = make([]float32, dModel*dModel)
		l.SelfKBias = make([]float32, dModel)
		l.SelfVWeight = make([]float32, dModel*dModel)
		l.SelfVBias = make([]float32, dModel)
		l.SelfOWeight = make([]float32, dModel*dModel)
		l.SelfOBias = make([]float32, dModel)
		l.CrossAttnLNWeight = ones(dModel)
		l.CrossAttnLNBias = make([]float32, dModel)
		l.CrossQWeight = make([]float32, dModel*dModel)
		l.CrossQBias = make([]float32, dModel)
		l.CrossKWeight = make([]float32, dModel*dModel)
		l.CrossKBias = make([]float32, dModel)
		l.CrossVWeight = make([]float32, dModel*dModel)
		l.CrossVBias = make([]float32, dModel)
		l.CrossOWeight = make([]float32, dModel*dModel)
		l.CrossOBias = make([]float32, dModel)
		l.MLPLNWeight = ones(dModel)
		l.MLPLNBias = make([]float32, dModel)
		l.FC1Weight = make([]float32, cfg.DecoderFFNDim*dModel)
		l.FC1Bias = make([]float32, cfg.DecoderFFNDim)
		l.FC2Weight = make([]float32, dModel*cfg.DecoderFFNDim)
		l.FC2Bias = make([]float32, dModel)
	}
	return dec
}
