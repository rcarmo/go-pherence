package whisper

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"math/rand"
	"reflect"
	"strings"
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

func TestPCMGenerationLimitSplitMappingAndFailures(t *testing.T) {
	samples := make([]float32, 1000)
	for i := range samples {
		samples[i] = float32(i + 1)
	}
	calls := 0
	segments, words, err := splitPCMGenerationLimit(context.Background(), samples, 801, func(padded []float32, valid int) ([]Segment, []WordTiming, error) {
		calls++
		wantValid := 400
		wantFirst := float32(1)
		if calls == 2 {
			wantValid, wantFirst = 401, 401
		}
		if valid != wantValid || padded[0] != wantFirst || padded[valid-1] == 0 || padded[valid] != 0 || len(padded) != len(samples) {
			t.Fatalf("split %d valid=%d first=%g tail=%g pad=%g", calls, valid, padded[0], padded[valid-1], padded[valid])
		}
		return []Segment{{Start: 0, End: float64(valid) / 16000, Text: "part", Tokens: []int{calls, calls + 10}}}, []WordTiming{{Word: "part", Start: 0, End: float64(valid) / 16000, TokenStart: 0, TokenEnd: 2}}, nil
	})
	if err != nil || calls != 2 || len(segments) != 2 || len(words) != 2 || segments[0].Start != 0 || segments[1].Start != 400.0/16000 || segments[1].End != 801.0/16000 || words[0].TokenStart != 0 || words[0].TokenEnd != 2 || words[1].TokenStart != 2 || words[1].TokenEnd != 4 {
		t.Fatal("split mapping", calls, segments, words, err)
	}
	// Only the failing first half is subdivided; the successful sibling is
	// decoded once, and all three leaves retain absolute time and word indices.
	calls = 0
	segments, words, err = splitPCMGenerationLimit(context.Background(), samples, 801, func(padded []float32, valid int) ([]Segment, []WordTiming, error) {
		calls++
		if padded[0] == 1 && valid == 400 {
			return nil, nil, ErrGenerationLimit
		}
		return []Segment{{Start: 0, End: float64(valid) / 16000, Tokens: []int{calls}}}, []WordTiming{{Word: "part", Start: 0, End: float64(valid) / 16000, TokenStart: 0, TokenEnd: 1}}, nil
	})
	if err != nil || calls != 4 || len(segments) != 3 || len(words) != 3 || segments[0].Start != 0 || segments[1].Start != 200.0/16000 || segments[2].Start != 400.0/16000 || words[2].TokenStart != 2 {
		t.Fatal("adaptive split", calls, segments, words, err)
	}
	calls = 0
	large := make([]float32, 480000)
	segments, words, err = splitPCMGenerationLimit(context.Background(), large, len(large), func([]float32, int) ([]Segment, []WordTiming, error) {
		calls++
		return nil, nil, ErrGenerationLimit
	})
	if !errors.Is(err, ErrGenerationLimit) || calls != maxPCMSplitDepth || segments != nil || words != nil {
		t.Fatal("unbounded split", calls, segments, words, err)
	}
	failure := errors.New("half failed")
	calls = 0
	segments, words, err = splitPCMGenerationLimit(context.Background(), samples, 801, func([]float32, int) ([]Segment, []WordTiming, error) {
		calls++
		if calls == 2 {
			return nil, nil, failure
		}
		return []Segment{{Start: 0, End: .01, Tokens: []int{1}}}, nil, nil
	})
	if !errors.Is(err, failure) || segments != nil || words != nil || calls != 2 {
		t.Fatal("partial split escaped", calls, segments, words, err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	calls = 0
	segments, words, err = splitPCMGenerationLimit(ctx, samples, 801, func([]float32, int) ([]Segment, []WordTiming, error) {
		calls++
		cancel()
		return nil, nil, nil
	})
	cancel()
	if !errors.Is(err, context.Canceled) || segments != nil || words != nil || calls != 1 {
		t.Fatal("split cancellation", calls, segments, words, err)
	}
	for _, tc := range []struct {
		ctx     context.Context
		samples []float32
		valid   int
		infer   func([]float32, int) ([]Segment, []WordTiming, error)
	}{{nil, samples, 801, func([]float32, int) ([]Segment, []WordTiming, error) { return nil, nil, nil }}, {context.Background(), samples, 319, func([]float32, int) ([]Segment, []WordTiming, error) { return nil, nil, nil }}, {context.Background(), samples, 1001, func([]float32, int) ([]Segment, []WordTiming, error) { return nil, nil, nil }}, {context.Background(), samples, 801, nil}} {
		if out, worded, err := splitPCMGenerationLimit(tc.ctx, tc.samples, tc.valid, tc.infer); !errors.Is(err, ErrGenerationLimit) || out != nil || worded != nil {
			t.Fatal("invalid split accepted", tc.valid, err)
		}
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
	_, zero, err := canonicalWindowOutput(w, nil, []WordTiming{{Word: "x", Start: .1, End: .1, TokenStart: 0, TokenEnd: 1}})
	if err != nil || len(zero) != 1 || zero[0].Start != zero[0].End {
		t.Fatal("checked zero-duration word rejected", zero, err)
	}
	for _, words := range [][]WordTiming{
		{{Word: "x", Start: -1, End: .1, TokenStart: 0, TokenEnd: 1}},
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

func TestPCMTranscribeAutomaticLanguage(t *testing.T) {
	t.Setenv("GO_PHERENCE_DISABLE_NVIDIA", "1")
	w := toyPCMModel()
	tok, raw := generationJSONFixture(t, w.Config)
	generation, err := ParseGenerationConfigChecked(marshalConfig(t, raw), w.Config, tok)
	if err != nil {
		t.Fatal(err)
	}
	// The decoder output projection shares TokenEmbed. Make Portuguese the
	// unique highest language logit without changing the timestamp path.
	w.Decoder.TokenEmbed[50267*2] = 20
	w.Decoder.TokenEmbed[50267*2+1] = -20
	reader := sampleReadFunc(func(_ context.Context, dst []float32, _ int64) (int, error) {
		clear(dst)
		return len(dst), nil
	})
	var got []WindowTranscript
	err = w.TranscribePCMWindows(context.Background(), reader, 321, tok, PCMTranscribeOptions{Language: "auto", Generation: generation}, func(out WindowTranscript) error {
		got = append(got, out)
		return nil
	})
	if err != nil || len(got) != 2 {
		t.Fatalf("auto windows=%d err=%v", len(got), err)
	}
	for _, window := range got {
		if window.Language != "pt" {
			t.Fatalf("language=%q", window.Language)
		}
	}
	fixed := WindowTranscript{Window: got[0].Window, Segments: []Segment{}}
	encoded, err := json.Marshal(fixed)
	if err != nil || strings.Contains(string(encoded), "Language") {
		t.Fatalf("fixed-language record changed: %s %v", encoded, err)
	}
}

func TestPCMTranscribeAutomaticLanguageRequiresGeneration(t *testing.T) {
	w := toyPCMModel()
	tok := checkedTestTokenizer(w.Config.VocabSize)
	reader := sampleReadFunc(func(context.Context, []float32, int64) (int, error) {
		t.Fatal("invalid auto mode reached source")
		return 0, nil
	})
	if err := w.TranscribePCMWindows(context.Background(), reader, 1, tok, PCMTranscribeOptions{Language: "auto"}, func(WindowTranscript) error { return nil }); err == nil {
		t.Fatal("automatic language accepted without generation metadata")
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

// Original window compatibility carries a rolling prompt across windows, so a
// resume must replay earlier windows silently and emit exactly what an
// uninterrupted run emits from the first missing window onward.
func TestPCMOriginalWindowCompatibilityResumeReplays(t *testing.T) {
	t.Setenv("GO_PHERENCE_DISABLE_NVIDIA", "1")
	w := toyPCMModel()
	tok := checkedTestTokenizer(w.Config.VocabSize)
	tok.Vocab[w.Config.VocabSize-1506+2] = "<|startofprev|>" // transcribe+2, as in the multilingual layout
	r := rand.New(rand.NewSource(3))
	pcm := make([]float32, 12*320+7) // > delta_min (100 ms) beyond several windows
	for i := range pcm {
		pcm[i] = float32(r.NormFloat64() * 0.1)
	}
	reads := map[int64]int{}
	source := sampleReadFunc(func(_ context.Context, dst []float32, start int64) (int, error) {
		reads[start]++
		n := copy(dst, pcm[min(int(start), len(pcm)):])
		for i := n; i < len(dst); i++ {
			dst[i] = 0
		}
		return len(dst), nil
	})
	opts := PCMTranscribeOptions{Language: "pt", OriginalDecoderCompatibility: true, OriginalWindowCompatibility: true}
	run := func(first int64) []WindowTranscript {
		clear(reads)
		var out []WindowTranscript
		if e := w.TranscribePCMWindowsFrom(context.Background(), source, int64(len(pcm)), tok, opts, first, func(r WindowTranscript) error { out = append(out, r); return nil }); e != nil {
			t.Fatal(first, e)
		}
		return out
	}
	full := run(0)
	if len(full) < 4 {
		t.Fatal("windows", len(full))
	}
	for _, first := range []int64{1, 3, int64(len(full) - 1)} {
		got := run(first)
		if !reflect.DeepEqual(got, full[first:]) {
			t.Fatalf("resume %d differs: %+v vs %+v", first, got, full[first:])
		}
		// Every window is read by the whole-clip mel pass and again for inference,
		// including the silently replayed windows before first.
		for _, f := range full {
			if reads[f.Window.Start] < 2 {
				t.Fatalf("resume %d did not replay window %d (reads %d)", first, f.Window.Index, reads[f.Window.Start])
			}
		}
	}
	nonEmpty := 0
	for _, f := range full {
		if f.Prompt == nil || len(*f.Prompt) > maxPreviousTextTokens {
			t.Fatalf("window %d prompt %v", f.Window.Index, f.Prompt)
		}
		nonEmpty += min(len(*f.Prompt), 1)
	}
	if nonEmpty == 0 {
		t.Fatal("toy run carried no rolling prompt")
	}
	// Resuming from the persisted prompt of window first-1 skips inference of
	// earlier windows (read once, by the whole-clip mel pass) and emits the same.
	for _, first := range []int64{1, 3, int64(len(full) - 1)} {
		clear(reads)
		var got []WindowTranscript
		if e := w.TranscribePCMWindowsFromPrompt(context.Background(), source, int64(len(pcm)), tok, opts, first, WindowResume{Start: full[first-1].Window.EmitEnd, Prompt: *full[first-1].Prompt}, func(r WindowTranscript) error { got = append(got, r); return nil }); e != nil {
			t.Fatal(first, e)
		}
		if !reflect.DeepEqual(got, full[first:]) {
			t.Fatalf("prompt resume %d differs", first)
		}
		for _, f := range full[:first] {
			if reads[f.Window.Start] != 1 {
				t.Fatalf("prompt resume %d re-inferred window %d (reads %d)", first, f.Window.Index, reads[f.Window.Start])
			}
		}
	}
	for _, bad := range [][]int{make([]int, maxPreviousTextTokens+1), {-1}, {w.Config.VocabSize}} {
		if e := w.TranscribePCMWindowsFromPrompt(context.Background(), source, int64(len(pcm)), tok, opts, 1, WindowResume{Start: 320, Prompt: bad}, func(WindowTranscript) error { return nil }); e == nil {
			t.Fatal("invalid prompt accepted")
		}
	}
	if e := w.TranscribePCMWindowsFromPrompt(context.Background(), source, int64(len(pcm)), tok, opts, 0, WindowResume{Start: 320}, func(WindowTranscript) error { return nil }); e == nil {
		t.Fatal("prompt resume at window 0 accepted")
	}
}

func TestDecodeDroppingHistoryOnLimit(t *testing.T) {
	history := []int{1, 2, 3}
	calls := 0
	_, _, _, err := decodeDroppingHistoryOnLimit(&history, func() ([]Segment, []WordTiming, error) {
		calls++
		if len(history) > 0 {
			return nil, nil, ErrGenerationLimit
		}
		history = []int{9}
		return []Segment{{Text: "ok"}}, nil, nil
	})
	if err != nil || calls != 2 || len(history) != 1 || history[0] != 9 {
		t.Fatal(err, calls, history)
	}
	// No history: the limit is returned unchanged for split recovery.
	empty := []int{}
	calls = 0
	if _, _, _, err := decodeDroppingHistoryOnLimit(&empty, func() ([]Segment, []WordTiming, error) { calls++; return nil, nil, ErrGenerationLimit }); !errors.Is(err, ErrGenerationLimit) || calls != 1 {
		t.Fatal(err, calls)
	}
	// Non-limit errors and nil history (no window compatibility) never retry.
	calls = 0
	if _, _, _, err := decodeDroppingHistoryOnLimit(nil, func() ([]Segment, []WordTiming, error) { calls++; return nil, nil, ErrGenerationLimit }); !errors.Is(err, ErrGenerationLimit) || calls != 1 {
		t.Fatal(err, calls)
	}
	history = []int{1}
	calls = 0
	boom := errors.New("boom")
	if _, _, _, err := decodeDroppingHistoryOnLimit(&history, func() ([]Segment, []WordTiming, error) { calls++; return nil, nil, boom }); err != boom || calls != 1 || len(history) != 1 {
		t.Fatal(err, calls)
	}
}

func TestDecodeRetriesRepetitiveGenerationWithoutHistory(t *testing.T) {
	loop := make([]int, 0, 40)
	for len(loop) < 40 {
		loop = append(loop, 7, 8) // entropy ln2 < 2.4
	}
	varied := make([]int, 40)
	for i := range varied {
		varied[i] = 100 + i // entropy ln32 > 2.4
	}
	if !repetitiveGeneration(loop) || repetitiveGeneration(varied) || repetitiveGeneration(loop[:32]) {
		t.Fatal("entropy test")
	}
	history := []int{1, 2, 3}
	calls := 0
	_, _, _, err := decodeDroppingHistoryOnLimit(&history, func() ([]Segment, []WordTiming, error) {
		calls++
		if len(history) > 0 {
			history = append(history, loop...) // decodePCMWindow appends generated tokens
			return []Segment{{Text: "loop"}}, nil, nil
		}
		history = append(history, varied...)
		return []Segment{{Text: "ok"}}, nil, nil
	})
	if err != nil || calls != 2 || len(history) != len(varied) {
		t.Fatal(err, calls, len(history))
	}
	// A varied generation with history is kept after one decode.
	history = []int{1, 2, 3}
	calls = 0
	if _, _, _, err := decodeDroppingHistoryOnLimit(&history, func() ([]Segment, []WordTiming, error) {
		calls++
		history = append(history, varied...)
		return nil, nil, nil
	}); err != nil || calls != 1 {
		t.Fatal(err, calls)
	}
}

func TestHighCompressionRatioFlagsRepeatedText(t *testing.T) {
	loop := make([]Segment, 9)
	for i := range loop {
		loop[i] = Segment{Text: " So I was able to do this in the next phase."}
	}
	speech := []Segment{{Text: " And so, my fellow Americans, ask not what your country can do for you,"}, {Text: " ask what you can do for your country."}}
	if !highCompressionRatio(loop) || highCompressionRatio(speech) || highCompressionRatio(nil) {
		t.Fatal("compression ratio test")
	}
	history := []int{1, 2, 3}
	calls := 0
	segments, _, _, err := decodeDroppingHistoryOnLimit(&history, func() ([]Segment, []WordTiming, error) {
		calls++
		if len(history) > 0 {
			return loop, nil, nil
		}
		return speech, nil, nil
	})
	if err != nil || calls != 2 || segments[0].Text != speech[0].Text {
		t.Fatal(err, calls)
	}
}

func TestRepeatedSegments(t *testing.T) {
	seg := func(texts ...string) []Segment {
		out := make([]Segment, len(texts))
		for i, text := range texts {
			out[i] = Segment{Text: text}
		}
		return out
	}
	if !repeatedSegments(seg(" a", " x.", " x.", "x. ", " b")) || repeatedSegments(seg(" x.", " x.", " y", " x.")) || repeatedSegments(seg("", "", "")) || repeatedSegments(nil) {
		t.Fatal("repeated segment test")
	}
}

func TestWhisperSeekDeltaMatchesWhisperCpp(t *testing.T) {
	const beg, text = 1000, 5
	for _, c := range []struct {
		generated []int
		want      int64
	}{
		{nil, 3000},                                      // no timestamps: full chunk
		{[]int{beg, text, beg + 100}, 3000},              // text then single timestamp: last segment closed
		{[]int{beg, text, beg + 100, beg + 120}, 240},    // dangling start: seek to it
		{[]int{beg + 3, text, beg + 50, beg}, 100},       // <|0.00|> never moves the seek
		{[]int{beg, text, beg + 1500, beg + 1500}, 3000}, // 30 s timestamp
	} {
		if got := whisperSeekDelta(c.generated, beg); got != c.want {
			t.Fatal(c.generated, got, c.want)
		}
	}
}

func TestPCMSeekWindowsChainOnLastTimestamp(t *testing.T) {
	total := int64(75 * SpeechSampleRate)
	source := sampleReadFunc(func(_ context.Context, dst []float32, start int64) (int, error) {
		for i := range dst {
			dst[i] = float32((start+int64(i))%97) / 97
		}
		return len(dst), nil
	})
	var current, delta int64
	deltas := []int64{2000, 3000, 1234}
	var got []Window
	calls := 0
	err := transcribePCMSeekFrom(context.Background(), source, total, MaxWindowSamples, 0, 0, &current, &delta, func(w WindowTranscript) error { got = append(got, w.Window); return nil }, func(samples []float32, valid int) ([]Segment, []WordTiming, error) {
		delta = deltas[min(calls, len(deltas)-1)]
		calls++
		return nil, nil, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	start := int64(0)
	for i, w := range got {
		if !ValidSeekWindow(w, int64(i), start, total, MaxWindowSamples) {
			t.Fatalf("window %d invalid: %+v", i, w)
		}
		start = w.EmitEnd
	}
	// 0→20 s (seek 2000), →50 s (full chunk), then 12.34 s steps until done.
	if got[0].EmitEnd != 2000*160 || got[1].Start != 2000*160 || got[1].EmitEnd != 5000*160 || !WhisperSeekDone(start, total) {
		t.Fatalf("%+v", got)
	}
	if ValidSeekWindow(got[1], 1, 0, total, MaxWindowSamples) || ValidSeekWindow(Window{Index: 0, Start: total - 800, End: total, EmitStart: total - 800, EmitEnd: total, InputSamples: MaxWindowSamples, PadSamples: MaxWindowSamples - 800}, 0, total-800, total, MaxWindowSamples) {
		t.Fatal("invalid seek window accepted")
	}
}

// Adjacent seek windows must agree on shared boundaries exactly: window 32 of
// the podcast ended at 704.94+17.82 s, which float addition made 722.7600000000001
// while the next window started at 722.76 s, a false transcript overlap.
func TestCanonicalWindowOutputSampleExactBoundaries(t *testing.T) {
	first := Window{Index: 32, Start: 11279040, End: 11279040 + MaxWindowSamples, EmitStart: 11279040, EmitEnd: 11564160, InputSamples: MaxWindowSamples}
	second := Window{Index: 33, Start: 11564160, End: 11564160 + MaxWindowSamples, EmitStart: 11564160, EmitEnd: 11564160 + MaxWindowSamples, InputSamples: MaxWindowSamples}
	a, _, err := canonicalWindowOutput(first, []Segment{{Start: float64(703) / 50, End: float64(891) / 50, Text: "a"}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	b, _, err := canonicalWindowOutput(second, []Segment{{Start: 0, End: 1, Text: "b"}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if a[0].End != b[0].Start || a[0].End != 722.76 {
		t.Fatalf("%v != %v", a[0].End, b[0].Start)
	}
}
