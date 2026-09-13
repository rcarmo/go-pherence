package whisper

import (
	"context"
	"errors"
	"math"
	"testing"
	"time"
)

func TestPCMTranscribePlanAllWindowsAndFailures(t *testing.T) {
	// >100 windows, one frame in the final tail; cheap synthetic infer stub.
	plan, _ := NewWindowPlan(160*130+1, 160, 0)
	reads, inferences, emits := 0, 0, 0
	var scratchBase *float32
	source := sampleReadFunc(func(_ context.Context, dst []float32, start int64) (int, error) {
		if start != int64(reads)*160 {
			t.Fatalf("lost sample offset %d", start)
		}
		reads++
		for i := range dst {
			dst[i] = 0.5
		}
		return len(dst), nil
	})
	infer := func(samples []float32, _ int) ([]Segment, []WordTiming, error) {
		inferences++
		if scratchBase == nil {
			scratchBase = &samples[0]
		} else if scratchBase != &samples[0] {
			t.Fatal("new scratch per window")
		}
		if inferences == 131 && (samples[0] != 0.5 || samples[1] != 0) {
			t.Fatal("final-tail padding")
		}
		return nil, nil, nil
	}
	if err := transcribePCMPlan(context.Background(), source, plan, func(r WindowTranscript) error {
		if r.Window.Index != int64(emits) {
			t.Fatal("callback order")
		}
		emits++
		return nil
	}, infer); err != nil {
		t.Fatal(err)
	}
	if reads != 131 || inferences != 131 || emits != 131 {
		t.Fatalf("cutoff %d/%d/%d", reads, inferences, emits)
	}
	empty, _ := NewWindowPlan(0, 160, 0)
	if err := transcribePCMPlan(context.Background(), nil, empty, nil, nil); err != nil {
		t.Fatal(err)
	}
	// Each stage error aborts, without a failed-window callback or later read.
	failure := errors.New("stage failure")
	for _, stage := range []string{"read", "infer", "emit", "cancel"} {
		t.Run(stage, func(t *testing.T) {
			reads, emits := 0, 0
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			s := sampleReadFunc(func(_ context.Context, dst []float32, _ int64) (int, error) {
				reads++
				if stage == "read" {
					return 0, failure
				}
				return len(dst), nil
			})
			infer := func([]float32, int) ([]Segment, []WordTiming, error) {
				if stage == "infer" {
					return nil, nil, failure
				}
				if stage == "cancel" {
					cancel()
				}
				return nil, nil, nil
			}
			err := transcribePCMPlan(ctx, s, plan, func(WindowTranscript) error { emits++; return failure }, infer)
			if reads != 1 || ((stage == "emit") != (emits == 1)) {
				t.Fatal("incorrect partial checkpoint boundary")
			}
			if stage == "cancel" {
				if !errors.Is(err, context.Canceled) {
					t.Fatal(err)
				}
			} else if !errors.Is(err, failure) {
				t.Fatal(err)
			}
		})
	}
}

func TestPCMTranscribeTimestampMapping(t *testing.T) {
	p, _ := NewWindowPlan(480001, 480000, 16000)
	w, _ := p.At(1)
	segs, err := canonicalWindowSegments(w, []Segment{{Start: 0, End: 0.5, Text: "a"}, {Start: 0.5, End: 2, Text: "b"}, {Start: 2, End: 3, Text: "padding"}})
	if err != nil {
		t.Fatal(err)
	}
	if len(segs) != 2 || segs[0].Start != 29 || segs[0].End != 29.5 || segs[1].End != float64(480001)/16000 {
		t.Fatalf("bad canonical times %+v", segs)
	}
	_, worded, err := canonicalWindowOutput(w, []Segment{{Start: 0, End: 1, Text: "two words", Tokens: []int{1, 2}}}, []WordTiming{{Word: "two", Start: .1, End: .4, TokenStart: 0, TokenEnd: 1}, {Word: "words", Start: .4, End: .9, TokenStart: 1, TokenEnd: 2}})
	if err != nil || len(worded) != 2 || worded[0].Start != 29.1 || worded[1].End != 29.9 {
		t.Fatal("word timestamp mapping", worded, err)
	}
	for _, segs := range [][]Segment{
		{{Start: -1, End: 1}}, {{Start: 1, End: 1}}, {{Start: 0, End: 31}}, {{Start: 0, End: math.Inf(1)}}, {{Start: math.NaN(), End: 1}}, {{Start: 0, End: 1}, {Start: 0.5, End: 2}},
	} {
		if _, err := canonicalWindowSegments(w, segs); err == nil {
			t.Fatalf("accepted malformed timestamps %+v", segs)
		}
	}
	for _, words := range [][]WordTiming{
		{{Word: "x", Start: -1, End: .1, TokenStart: 0, TokenEnd: 1}},
		{{Word: "x", Start: .1, End: .1, TokenStart: 0, TokenEnd: 1}},
		{{Word: "x", Start: .1, End: 2, TokenStart: 0, TokenEnd: 1}},
		{{Word: "x", Start: .1, End: .3, TokenStart: 1, TokenEnd: 2}},
		{{Word: " ", Start: .1, End: .3, TokenStart: 0, TokenEnd: 1}},
		{{Word: "x", Start: .1, End: .4, TokenStart: 0, TokenEnd: 1}, {Word: "y", Start: .3, End: .5, TokenStart: 1, TokenEnd: 2}},
	} {
		if _, _, err := canonicalWindowOutput(w, nil, words); err == nil {
			t.Fatalf("accepted malformed word timestamps %+v", words)
		}
	}
}

// Two-channel, zero-layer toy tensors exercise the actual checked frontend,
// encoder and decoder code. This is NOT a checkpoint or model-quality test.
func toyPCMModel() *Whisper {
	c := Config{NumMelBins: 80, MaxLength: 2, EncoderDModel: 2, DecoderDModel: 2, EncoderHeads: 1, DecoderHeads: 1, EncoderFFNDim: 2, DecoderFFNDim: 2, HeadDim: 2, VocabSize: 51865, MaxDecoderLength: 8}
	e, d := NewEncoder(c), NewDecoder(c)
	e.Conv1Weight = make([]float32, 2*80*3)
	e.Conv1Bias = make([]float32, 2)
	e.Conv2Weight = make([]float32, 2*2*3)
	e.Conv2Bias = make([]float32, 2)
	e.FinalLNWeight = []float32{1, 1}
	e.FinalLNBias = make([]float32, 2)
	d.TokenEmbed = make([]float32, c.VocabSize*2)
	d.TokenEmbed[50257*2] = 10
	d.TokenEmbed[50257*2+1] = -10
	d.PosEmbed = make([]float32, c.MaxDecoderLength*2)
	for i := 0; i < c.MaxDecoderLength; i++ {
		d.PosEmbed[2*i] = 1
		d.PosEmbed[2*i+1] = -1
	}
	d.FinalLNWeight = []float32{1, 1}
	d.FinalLNBias = make([]float32, 2)
	return &Whisper{Encoder: e, Decoder: d, Config: c}
}

func TestPCMTranscribeActualToyPipeline(t *testing.T) {
	t.Setenv("GO_PHERENCE_DISABLE_NVIDIA", "1")
	w := toyPCMModel()
	tok := checkedTestTokenizer(w.Config.VocabSize)
	calls, emits := 0, 0
	reader := sampleReadFunc(func(_ context.Context, dst []float32, _ int64) (int, error) {
		calls++
		clear(dst)
		return len(dst), nil
	})
	err := w.TranscribePCMWindows(context.Background(), reader, 321, tok, PCMTranscribeOptions{Language: "pt"}, func(out WindowTranscript) error {
		emits++
		if len(out.Segments) != 0 {
			t.Fatal("toy EOT expected no text")
		}
		return nil
	})
	if err != nil || calls != 2 || emits != 2 {
		t.Fatalf("toy pipeline %d/%d %v", calls, emits, err)
	}
}

func TestPCMTranscribeValidationAndGate(t *testing.T) {
	w := toyPCMModel()
	tok := checkedTestTokenizer(w.Config.VocabSize)
	reader := sampleReadFunc(func(context.Context, []float32, int64) (int, error) {
		t.Fatal("validation failure reached source")
		return 0, nil
	})
	emit := func(WindowTranscript) error { t.Fatal("validation failure reached output"); return nil }
	cases := []struct {
		total int64
		opts  PCMTranscribeOptions
	}{
		{-1, PCMTranscribeOptions{Language: "pt"}}, {4*3600*16000 + 1, PCMTranscribeOptions{Language: "pt"}},
		{1, PCMTranscribeOptions{}}, {1, PCMTranscribeOptions{Language: "pt", OverlapSamples: 320}},
		{1, PCMTranscribeOptions{Language: "pt", OverlapSamples: 161}},
		{320*10000 + 1, PCMTranscribeOptions{Language: "pt"}},
		{1, PCMTranscribeOptions{Language: "pt", MaxNewTokens: 6}}, {1, PCMTranscribeOptions{Language: "pt", MaxNewTokens: -1}},
		{1, PCMTranscribeOptions{Language: "pt", MaxInitialTimestampIndex: 1501}},
	}
	for _, c := range cases {
		if err := w.TranscribePCMWindows(context.Background(), reader, c.total, tok, c.opts, emit); err == nil {
			t.Fatalf("accepted %+v", c)
		}
	}
	w.Decoder.TokenEmbed = nil
	if err := w.TranscribePCMWindows(context.Background(), reader, 1, tok, PCMTranscribeOptions{Language: "pt"}, emit); err == nil {
		t.Fatal("missing weights accepted")
	}
	pcmInferenceGate <- struct{}{}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()
	err := w.TranscribePCMWindows(ctx, reader, 1, tok, PCMTranscribeOptions{Language: "pt"}, emit)
	<-pcmInferenceGate
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatal("cancel while waiting", err)
	}
	var missing *Whisper
	if missing.validatePCMModel() == nil {
		t.Fatal("nil model accepted")
	}
	for _, mutate := range []func(*Whisper){
		func(w *Whisper) { w.Config.EncoderHeads = 0 }, func(w *Whisper) { w.Config.EncoderDModel = math.MaxInt },
		func(w *Whisper) { w.Encoder.PosEmbed = nil }, func(w *Whisper) { w.Decoder.cfg.MaxLength++ }, func(w *Whisper) { w.Encoder.Conv1Weight = nil },
	} {
		w = toyPCMModel()
		mutate(w)
		if w.validatePCMModel() == nil {
			t.Fatal("malformed model accepted")
		}
	}
}

func TestPCMResumePlanKeepsAbsoluteGeometry(t *testing.T) {
	plan, _ := NewWindowPlan(1001, 320, 160)
	var offsets []int64
	source := sampleReadFunc(func(_ context.Context, dst []float32, start int64) (int, error) {
		offsets = append(offsets, start)
		return len(dst), nil
	})
	var results []WindowTranscript
	infer := func([]float32, int) ([]Segment, []WordTiming, error) {
		return []Segment{{Start: 0, End: 0.01, Text: "synthetic"}}, nil, nil
	}
	if e := transcribePCMPlanFrom(context.Background(), source, plan, 2, func(w WindowTranscript) error { results = append(results, w); return nil }, infer); e != nil {
		t.Fatal(e)
	}
	if len(results) != int(plan.Count()-2) || offsets[0] != 320 {
		t.Fatal(offsets, results)
	}
	for i, r := range results {
		want, _ := plan.At(int64(i + 2))
		if r.Window != want || r.Segments[0].Start != float64(want.Start)/16000 {
			t.Fatal(r, want)
		}
	}
	for _, first := range []int64{-1, plan.Count() + 1} {
		if e := transcribePCMPlanFrom(context.Background(), source, plan, first, nil, nil); e == nil {
			t.Fatal("invalid prefix")
		}
	}
	before := len(offsets)
	if e := transcribePCMPlanFrom(context.Background(), source, plan, plan.Count(), nil, nil); e != nil || len(offsets) != before {
		t.Fatal("completed prefix reread", e)
	}
}

func TestPCMResumeActualToyAndHostOnly(t *testing.T) {
	t.Setenv("GO_PHERENCE_DISABLE_NVIDIA", "1")
	t.Setenv("GO_PHERENCE_WHISPER_GPU_GRAPH", "0")
	t.Setenv("GO_PHERENCE_WHISPER_GPU_SELF_ATTN", "0")
	w := toyPCMModel()
	tok := checkedTestTokenizer(w.Config.VocabSize)
	if e := w.ValidatePCMHostOnly(); e != nil {
		t.Fatal(e)
	}
	calls := 0
	source := sampleReadFunc(func(_ context.Context, dst []float32, start int64) (int, error) {
		calls++
		if start != 320 {
			t.Fatal("resumed wrong window", start)
		}
		return len(dst), nil
	})
	if e := w.TranscribePCMWindowsFrom(context.Background(), source, 321, tok, PCMTranscribeOptions{Language: "pt"}, 1, func(r WindowTranscript) error {
		if r.Window.Index != 1 || r.Window.Start != 320 {
			t.Fatal(r)
		}
		return nil
	}); e != nil || calls != 1 {
		t.Fatal(calls, e)
	}
	t.Setenv("GO_PHERENCE_WHISPER_GPU_SELF_ATTN", "1")
	if e := w.ValidatePCMHostOnly(); e == nil {
		t.Fatal("GPU feature accepted")
	}
	t.Setenv("GO_PHERENCE_WHISPER_GPU_SELF_ATTN", "0")
	t.Setenv("GO_PHERENCE_DISABLE_NVIDIA", "0")
	if e := w.ValidatePCMHostOnly(); e == nil {
		t.Fatal("NVIDIA not disabled")
	}
}
