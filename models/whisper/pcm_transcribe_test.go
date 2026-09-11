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
	infer := func(samples []float32) ([]Segment, error) {
		inferences++
		if scratchBase == nil {
			scratchBase = &samples[0]
		} else if scratchBase != &samples[0] {
			t.Fatal("new scratch per window")
		}
		if inferences == 131 && (samples[0] != 0.5 || samples[1] != 0) {
			t.Fatal("final-tail padding")
		}
		return nil, nil
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
			infer := func([]float32) ([]Segment, error) {
				if stage == "infer" {
					return nil, failure
				}
				if stage == "cancel" {
					cancel()
				}
				return nil, nil
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
	for _, segs := range [][]Segment{
		{{Start: -1, End: 1}}, {{Start: 1, End: 1}}, {{Start: 0, End: 31}}, {{Start: 0, End: math.Inf(1)}}, {{Start: math.NaN(), End: 1}}, {{Start: 0, End: 1}, {Start: 0.5, End: 2}},
	} {
		if _, err := canonicalWindowSegments(w, segs); err == nil {
			t.Fatalf("accepted malformed timestamps %+v", segs)
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
