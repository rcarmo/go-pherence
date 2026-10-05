package whisper

import (
	"context"
	"errors"
	"fmt"
	"math"
	"reflect"
	"testing"

	assets "github.com/rcarmo/go-pherence/loader/silero"
	vad "github.com/rcarmo/go-pherence/model/silero"
)

func toyVADModel(t *testing.T, speech bool) *vad.Model {
	t.Helper()
	f := &assets.File{Version: [3]int{6, 2, 0}, Window: 512, Context: 64, Tensors: map[string]assets.Tensor{}}
	put := func(name string, shape ...int) {
		n := 1
		for _, d := range shape {
			n *= d
		}
		f.Tensors[name] = assets.Tensor{Shape: shape, Values: make([]float32, n)}
	}
	put("_model.stft.forward_basis_buffer", 256, 1, 258)
	channels := [5]int{129, 128, 64, 64, 128}
	for i := 0; i < 4; i++ {
		p := fmt.Sprintf("_model.encoder.%d.reparam_conv", i)
		put(p+".weight", 3, channels[i], channels[i+1])
		put(p+".bias", channels[i+1])
	}
	put("_model.decoder.rnn.weight_ih", 128, 512)
	put("_model.decoder.rnn.weight_hh", 128, 512)
	put("_model.decoder.rnn.bias_ih", 512)
	put("_model.decoder.rnn.bias_hh", 512)
	put("_model.decoder.decoder.2.weight", 128)
	put("_model.decoder.decoder.2.bias")
	bias := float32(-4)
	if speech {
		bias = 4
	}
	f.Tensors["_model.decoder.decoder.2.bias"].Values[0] = bias
	m, err := vad.New(f)
	if err != nil {
		t.Fatal(err)
	}
	return m
}
func TestCheckedPCMVADToySpeechAndSilence(t *testing.T) {
	t.Setenv("GO_PHERENCE_DISABLE_NVIDIA", "1")
	w := toyPCMModel()
	tok := checkedTestTokenizer(w.Config.VocabSize)
	settings := PCMVADOptions{AllowExperimental: true, Model: toyVADModel(t, true), Segmentation: vad.SegmentOptions{Threshold: 0.5, MinSilenceSamples: 1}}
	reader := &countingReader{}
	emits := 0
	err := w.TranscribePCMWindowsWithVAD(context.Background(), reader, 321, tok, PCMTranscribeOptions{Language: "pt", MaxNewTokens: 3}, settings, func(out VADWindowTranscript) error {
		emits++
		if len(out.Segments) != 0 || len(out.OriginalSpeech) != 1 || out.OriginalSpeech[0].Start != out.CompactedWindow.Start || out.OriginalSpeech[0].End != out.CompactedWindow.End {
			t.Fatal("speech window", out)
		}
		return nil
	})
	if err != nil || emits != 2 || reader.calls != 3 {
		t.Fatal("toy speech", emits, reader.calls, err)
	}
	settings.Model = toyVADModel(t, false)
	reader = &countingReader{}
	emits = 0
	err = w.TranscribePCMWindowsWithVAD(context.Background(), reader, 321, tok, PCMTranscribeOptions{Language: "pt", MaxNewTokens: 3}, settings, func(out VADWindowTranscript) error { emits++; return nil })
	if err != nil || emits != 0 || reader.calls != 1 {
		t.Fatal("silence must skip encoder/decoder", emits, reader.calls, err)
	}
	settings.Model = toyVADModel(t, true)
	reader = &countingReader{}
	failure := errors.New("emit failed")
	err = w.TranscribePCMWindowsWithVAD(context.Background(), reader, 321, tok, PCMTranscribeOptions{Language: "pt", MaxNewTokens: 3}, settings, func(VADWindowTranscript) error { return failure })
	if !errors.Is(err, failure) || reader.calls != 2 {
		t.Fatal("callback failure", reader.calls, err)
	}
}
func TestCheckedPCMVADValidationBeforeReading(t *testing.T) {
	t.Setenv("GO_PHERENCE_DISABLE_NVIDIA", "1")
	w := toyPCMModel()
	tok := checkedTestTokenizer(w.Config.VocabSize)
	base := PCMVADOptions{AllowExperimental: true, Model: toyVADModel(t, true), Segmentation: vad.DefaultSegmentOptions()}
	emit := func(VADWindowTranscript) error { return nil }
	for _, test := range []struct {
		opts     PCMTranscribeOptions
		settings PCMVADOptions
		total    int64
		ctx      context.Context
	}{
		{PCMTranscribeOptions{Language: "pt"}, func() PCMVADOptions { o := base; o.AllowExperimental = false; return o }(), 160, context.Background()},
		{PCMTranscribeOptions{Language: "pt"}, func() PCMVADOptions { o := base; o.Model = nil; return o }(), 160, context.Background()},
		{PCMTranscribeOptions{Language: "pt"}, func() PCMVADOptions { o := base; o.Segmentation.Threshold = 2; return o }(), 160, context.Background()},
		{PCMTranscribeOptions{Language: "pt"}, base, 159, context.Background()},
		{PCMTranscribeOptions{Language: "pt"}, base, 4*3600*16000 + 1, context.Background()},
		{PCMTranscribeOptions{Language: "xx"}, base, 160, context.Background()},
		{PCMTranscribeOptions{Language: "auto"}, base, 160, context.Background()},
		{PCMTranscribeOptions{Language: "pt", WordTimestamps: true}, base, 160, context.Background()},
		{PCMTranscribeOptions{Language: "pt", MaxNewTokens: -1}, base, 160, context.Background()},
		{PCMTranscribeOptions{Language: "pt", MaxInitialTimestampIndex: 1501}, base, 160, context.Background()},
		{PCMTranscribeOptions{Language: "pt", OverlapSamples: -1}, base, 160, context.Background()},
		{PCMTranscribeOptions{Language: "pt", OverlapSamples: 1000000}, base, 160, context.Background()},
		{PCMTranscribeOptions{Language: "pt"}, base, 160, nil},
	} {
		r := &countingReader{}
		err := w.TranscribePCMWindowsWithVAD(test.ctx, r, test.total, tok, test.opts, test.settings, emit)
		if err == nil || r.calls != 0 {
			t.Fatal("invalid request did work", r.calls, err)
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	r := &countingReader{}
	if err := w.TranscribePCMWindowsWithVAD(ctx, r, 160, tok, PCMTranscribeOptions{Language: "pt"}, base, emit); !errors.Is(err, context.Canceled) || r.calls != 0 {
		t.Fatal(err)
	}
	if err := (*Whisper)(nil).TranscribePCMWindowsWithVAD(context.Background(), r, 160, tok, PCMTranscribeOptions{Language: "pt"}, base, emit); err == nil || r.calls != 0 {
		t.Fatal("nil model")
	}
	if err := w.TranscribePCMWindowsWithVAD(context.Background(), nil, 160, tok, PCMTranscribeOptions{Language: "pt"}, base, emit); err == nil {
		t.Fatal("nil source")
	}
	if err := w.TranscribePCMWindowsWithVAD(context.Background(), r, 160, tok, PCMTranscribeOptions{Language: "pt"}, base, nil); err == nil {
		t.Fatal("nil callback")
	}
}
func TestMapVADWindowOriginalWordAndSegmentBoundaries(t *testing.T) {
	spans := []vad.Span{{Start: 16000, End: 32000}, {Start: 80000, End: 96000}}
	reader, err := vad.NewCompactedReader(&countingReader{}, spans, 112000)
	if err != nil {
		t.Fatal(err)
	}
	window := WindowTranscript{Window: Window{Start: 0, End: 32000, EmitEnd: 32000, InputSamples: 32000}, Language: "pt", Segments: []Segment{{Start: 0, End: 1, Text: "first", Tokens: []int{1}}, {Start: 1, End: 2, Text: "second", Tokens: []int{2}}}, Words: []WordTiming{{Word: "first", Start: 0.1, End: 0.9, TokenStart: 0, TokenEnd: 1}, {Word: "second", Start: 1, End: 1.8, TokenStart: 1, TokenEnd: 2}}}
	out, err := mapVADWindow(window, reader, spans)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(out.OriginalSpeech, spans) || out.Language != "pt" || out.Segments[0].Start != 1 || out.Segments[0].End != 2 || out.Segments[1].Start != 5 || out.Segments[1].End != 6 || out.Words[0].Start != 1.1 || out.Words[1].Start != 5 {
		t.Fatal("original timeline", out)
	}
	window.Segments[0].Tokens[0] = 99
	if out.Segments[0].Tokens[0] != 1 {
		t.Fatal("tokens aliased")
	}
	// Exact compaction-boundary word start maps to the next speech interval.
	window.Words = []WordTiming{{Word: "boundary", Start: 1, End: 1, TokenStart: 0, TokenEnd: 1}}
	out, err = mapVADWindow(window, reader, spans)
	if err != nil || out.Words[0].Start != 5 || out.Words[0].End != 5 {
		t.Fatal("zero duration", out, err)
	}
	window.Words = []WordTiming{{Word: "crossing", Start: 0.9, End: 1.1, TokenStart: 0, TokenEnd: 1}}
	if _, err := mapVADWindow(window, reader, spans); err == nil {
		t.Fatal("word stretched across removed silence")
	}
	// Whole-segment envelope may enclose a gap, with retained ranges explicit.
	window.Words = nil
	window.Segments = []Segment{{Start: 0.5, End: 1.5, Text: "envelope"}}
	out, err = mapVADWindow(window, reader, spans)
	if err != nil || out.Segments[0].Start != 1.5 || out.Segments[0].End != 5.5 || out.Words != nil {
		t.Fatal(out, err)
	}
}
func TestMapVADWindowPartialRangeAndRejects(t *testing.T) {
	spans := []vad.Span{{Start: 16000, End: 32000}, {Start: 80000, End: 96000}}
	reader, _ := vad.NewCompactedReader(&countingReader{}, spans, 112000)
	window := WindowTranscript{Window: Window{Start: 8000, End: 24000, InputSamples: 16000}, Segments: []Segment{{Start: 0.5, End: 1.5, Text: "partial"}}}
	out, err := mapVADWindow(window, reader, spans)
	if err != nil || !reflect.DeepEqual(out.OriginalSpeech, []vad.Span{{Start: 24000, End: 32000}, {Start: 80000, End: 88000}}) {
		t.Fatal(out, err)
	}
	bad := window
	bad.Window.End = 33000
	if _, err := mapVADWindow(bad, reader, spans); err == nil {
		t.Fatal("bad window")
	}
	if _, err := mapVADWindow(window, nil, spans); err == nil {
		t.Fatal("nil mapping")
	}
	for _, segments := range [][]Segment{{{Start: 0.4, End: 1}}, {{Start: 1.4, End: 1.6}}, {{Start: 1, End: 1}}, {{Start: math.NaN(), End: 1}}, {{Start: 0.5, End: math.Inf(1)}}} {
		bad := window
		bad.Segments = segments
		if _, err := mapVADWindow(bad, reader, spans); err == nil {
			t.Fatal("bad segment")
		}
	}
	bad = window
	bad.Words = []WordTiming{{Word: "outside", Start: 0.4, End: 0.7}}
	if _, err := mapVADWindow(bad, reader, spans); err == nil {
		t.Fatal("bad word")
	}
}

type countingReader struct{ calls int }

func (r *countingReader) ReadSamplesAt(ctx context.Context, dst []float32, start int64) (int, error) {
	r.calls++
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	clear(dst)
	return len(dst), nil
}

func TestCheckedPCMVADAutoAndFailuresBeforePublish(t *testing.T) {
	t.Setenv("GO_PHERENCE_DISABLE_NVIDIA", "1")
	w := toyPCMModel()
	tok, raw := generationJSONFixture(t, w.Config)
	generation, err := ParseGenerationConfigChecked(marshalConfig(t, raw), w.Config, tok)
	if err != nil {
		t.Fatal(err)
	}
	w.Decoder.TokenEmbed[50267*2] = 100
	w.Decoder.TokenEmbed[50267*2+1] = -100
	settings := PCMVADOptions{AllowExperimental: true, Model: toyVADModel(t, true), Segmentation: vad.SegmentOptions{Threshold: 0.5, MinSilenceSamples: 1}}
	r := &countingReader{}
	emits := 0
	err = w.TranscribePCMWindowsWithVAD(context.Background(), r, 321, tok, PCMTranscribeOptions{Language: "auto", Generation: generation, MaxNewTokens: 3}, settings, func(out VADWindowTranscript) error {
		emits++
		if out.Language != "pt" {
			t.Fatal("detected language", out.Language)
		}
		return nil
	})
	if err != nil || emits != 2 {
		t.Fatal("auto VAD", emits, err)
	}
	sentinel := errors.New("VAD source failed")
	source := sampleReadFunc(func(context.Context, []float32, int64) (int, error) { return 0, sentinel })
	err = w.TranscribePCMWindowsWithVAD(context.Background(), source, 321, tok, PCMTranscribeOptions{Language: "pt"}, settings, func(VADWindowTranscript) error { t.Fatal("published failed VAD"); return nil })
	if !errors.Is(err, sentinel) {
		t.Fatal(err)
	}
	badGeneration := &CheckedGenerationConfig{}
	r = &countingReader{}
	err = w.TranscribePCMWindowsWithVAD(context.Background(), r, 321, tok, PCMTranscribeOptions{Language: "pt", Generation: badGeneration}, settings, func(VADWindowTranscript) error { return nil })
	if err == nil || r.calls != 0 {
		t.Fatal("invalid generation accepted before VAD", r.calls, err)
	}
	settings.Model = toyVADModel(t, false)
	err = w.TranscribePCMWindowsWithVAD(context.Background(), r, 321, tok, PCMTranscribeOptions{Language: "pt", VulkanEncoder: &VulkanEncoder{}}, settings, func(VADWindowTranscript) error { return nil })
	if err == nil || r.calls != 0 {
		t.Fatal("invalid Vulkan encoder did VAD", r.calls, err)
	}
}

func TestMapVADWindowRejectsMalformedWords(t *testing.T) {
	spans := []vad.Span{{Start: 16000, End: 32000}, {Start: 80000, End: 96000}}
	r, _ := vad.NewCompactedReader(&countingReader{}, spans, 112000)
	for _, words := range [][]WordTiming{
		{{Word: "", Start: 0.1, End: 0.2, TokenStart: 0, TokenEnd: 1}},
		{{Word: "x", Start: 0.1, End: 0.2, TokenStart: 1, TokenEnd: 2}},
		{{Word: "x", Start: 0.1, End: 0.2, TokenStart: 0, TokenEnd: 0}},
		{{Word: "x", Start: math.NaN(), End: 0.2, TokenStart: 0, TokenEnd: 1}},
		{{Word: "x", Start: 0.2, End: 0.1, TokenStart: 0, TokenEnd: 1}},
		{{Word: "x", Start: 0.1, End: 0.2, TokenStart: 0, TokenEnd: 1}, {Word: "y", Start: 0.15, End: 0.3, TokenStart: 1, TokenEnd: 2}},
	} {
		if _, err := mapVADWindow(WindowTranscript{Window: Window{Start: 0, End: 32000}, Words: words}, r, spans); err == nil {
			t.Fatal("malformed words")
		}
	}
}

func TestVADOriginalAudioGroupsPreserveGapEligibility(t *testing.T) {
	spans := []vad.Span{{Start: 100, End: 300}, {Start: 500, End: 700}, {Start: 1500, End: 3000}, {Start: 3200, End: 3500}}
	groups := groupVADAudioSpans(spans, 1000)
	want := []vad.Span{{Start: 100, End: 700}, {Start: 1500, End: 3000}, {Start: 3200, End: 3500}}
	if !reflect.DeepEqual(groups, want) || spans[0].End != 300 {
		t.Fatal("group ownership/extent", groups, spans)
	}
	speech := intersectVADSpans(spans, []vad.Span{{Start: 200, End: 600}, {Start: 2600, End: 3400}})
	if !reflect.DeepEqual(speech, []vad.Span{{Start: 200, End: 300}, {Start: 500, End: 600}, {Start: 2600, End: 3000}, {Start: 3200, End: 3400}}) {
		t.Fatal("eligibility intersection", speech)
	}
	source := &countingReader{}
	reader, err := vad.NewCompactedReader(source, groups[:1], 1000)
	if err != nil {
		t.Fatal(err)
	}
	// Actual retained silence now supplies the word alignment; no interpolation.
	window := WindowTranscript{Window: Window{Index: 0, Start: 0, End: 600}, Language: "en", Segments: []Segment{{Start: 0, End: 600.0 / 16000, Tokens: []int{1}}}, Words: []WordTiming{{Word: "real", Start: 100.0 / 16000, End: 500.0 / 16000, TokenStart: 0, TokenEnd: 1}}}
	got, err := mapVADWindow(window, reader, groups[:1])
	if err != nil {
		t.Fatal(err)
	}
	if got.Words[0].Start != 200.0/16000 || got.Words[0].End != 600.0/16000 {
		t.Fatal("actual audio word", got.Words)
	}
}

func TestPCMVADPreserveWindowGapsIsExplicit(t *testing.T) {
	ctx := context.Background()
	t.Setenv("GO_PHERENCE_DISABLE_NVIDIA", "1")
	w := toyPCMModel()
	tok := checkedTestTokenizer(w.Config.VocabSize)
	source := &countingReader{}
	emits := int64(0)
	err := w.TranscribePCMWindowsWithVAD(ctx, source, 321, tok, PCMTranscribeOptions{Language: "pt", MaxNewTokens: 3}, PCMVADOptions{Model: toyVADModel(t, true), Segmentation: vad.SegmentOptions{Threshold: .5, MinSilenceSamples: 1}, AllowExperimental: true, PreserveWindowGaps: true}, func(out VADWindowTranscript) error {
		if out.CompactedWindow.Index != emits || len(out.OriginalAudio) != 1 || !reflect.DeepEqual(out.OriginalSpeech, out.OriginalAudio) {
			t.Fatal("original window identity", out)
		}
		emits++
		return nil
	})
	if err != nil || emits != 2 {
		t.Fatal("retainedgap mode", err, emits)
	}
}

func TestVADGroupLongWindowCoordinates(t *testing.T) {
	spans := []vad.Span{{Start: 100, End: 600}, {Start: 1600, End: 1800}, {Start: 1900, End: 2100}}
	groups := groupVADAudioSpans(spans, 320)
	if !reflect.DeepEqual(groups, []vad.Span{{Start: 100, End: 600}, {Start: 1600, End: 1800}, {Start: 1900, End: 2100}}) {
		t.Fatal(groups)
	}
	// Independent retained original-index oracle checks that every group/window
	// is continuous and global retained-audio coordinates never reset.
	var offset, index int64
	for _, group := range groups {
		r, err := vad.NewCompactedReader(&countingReader{}, []vad.Span{group}, 2200)
		if err != nil {
			t.Fatal(err)
		}
		p, err := NewWindowPlan(r.Samples(), 320, 80)
		if err != nil {
			t.Fatal(err)
		}
		for i := int64(0); i < p.Count(); i++ {
			window, err := p.At(i)
			if err != nil {
				t.Fatal(err)
			}
			out, err := mapVADWindow(WindowTranscript{Window: window}, r, []vad.Span{group})
			if err != nil {
				t.Fatal(err)
			}
			out.CompactedWindow.Index = index
			out.CompactedWindow.Start += offset
			out.CompactedWindow.End += offset
			if out.CompactedWindow.Start != offset+window.Start || out.OriginalSpeech[0].Start != group.Start+window.Start || out.OriginalSpeech[0].End != group.Start+window.End {
				t.Fatal("retained coords", out, offset)
			}
			index++
		}
		offset += r.Samples()
	}
}

func TestPCMVADPreserveGapsCancelCallbackAndFreshRetry(t *testing.T) {
	t.Setenv("GO_PHERENCE_DISABLE_NVIDIA", "1")
	w := toyPCMModel()
	tok := checkedTestTokenizer(w.Config.VocabSize)
	settings := PCMVADOptions{Model: toyVADModel(t, true), AllowExperimental: true, PreserveWindowGaps: true, Segmentation: vad.SegmentOptions{Threshold: .5, MinSilenceSamples: 1}}
	ctx, cancel := context.WithCancel(context.Background())
	calls := 0
	err := w.TranscribePCMWindowsWithVAD(ctx, &countingReader{}, 321, tok, PCMTranscribeOptions{Language: "pt", MaxNewTokens: 3}, settings, func(out VADWindowTranscript) error { calls++; cancel(); return nil })
	if !errors.Is(err, context.Canceled) || calls != 1 {
		t.Fatal("cancel after callback", calls, err)
	}
	calls = 0
	var kept []VADWindowTranscript
	err = w.TranscribePCMWindowsWithVAD(context.Background(), &countingReader{}, 321, tok, PCMTranscribeOptions{Language: "pt", MaxNewTokens: 3}, settings, func(out VADWindowTranscript) error { calls++; kept = append(kept, out); return nil })
	if err != nil || calls != 2 {
		t.Fatal("fresh retry", calls, err)
	}
	if kept[0].OriginalAudio[0].End != 320 || kept[1].OriginalAudio[0].Start != 320 {
		t.Fatal("retry original windows", kept)
	}
	failure := errors.New("stop callback")
	calls = 0
	err = w.TranscribePCMWindowsWithVAD(context.Background(), &countingReader{}, 321, tok, PCMTranscribeOptions{Language: "pt", MaxNewTokens: 3}, settings, func(out VADWindowTranscript) error { calls++; return failure })
	if !errors.Is(err, failure) || calls != 1 {
		t.Fatal("callback failure", calls, err)
	}
}
