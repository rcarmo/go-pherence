package silero

import (
	"math"
	"reflect"
	"testing"
)

func TestSpeechSpansOriginalPaddingAndSilence(t *testing.T) {
	opts := DefaultSegmentOptions()
	probabilities := make([]float32, 80)
	for i := 10; i < 25; i++ {
		probabilities[i] = 0.9
	}
	for i := 50; i < 68; i++ {
		probabilities[i] = 0.8
	}
	got, err := SpeechSpans(probabilities, 80*512, opts)
	if err != nil {
		t.Fatal(err)
	}
	want := []Span{{10*512 - 480, 25*512 + 480}, {50*512 - 480, 68*512 + 480}}
	if !reflect.DeepEqual(got, want) {
		t.Fatal(got, want)
	}
	if got, err := SpeechSpans(make([]float32, 20), 20*512, opts); err != nil || len(got) != 0 {
		t.Fatal("silence", got, err)
	}
	for i := range probabilities {
		probabilities[i] = 0.9
	}
	if got, err := SpeechSpans(probabilities, 80*512-300, opts); err != nil || !reflect.DeepEqual(got, []Span{{0, 80*512 - 300}}) {
		t.Fatal("tail bounds", got, err)
	}
}
func TestSpeechSpansHysteresisMergeAndShortIslands(t *testing.T) {
	opts := DefaultSegmentOptions()
	probabilities := make([]float32, 55)
	for i := 0; i < 10; i++ {
		probabilities[i] = 0.6
	}
	// Four-window silence closes the span;200msmergejoins the next speech.
	for i := 15; i < 26; i++ {
		probabilities[i] = 0.7
	}
	// Probabilities between0.35..0.5 sustain a triggered span.
	for i := 26; i < 30; i++ {
		probabilities[i] = 0.4
	}
	// Short island (96ms) later must be rejected.
	for i := 46; i < 49; i++ {
		probabilities[i] = 0.9
	}
	got, err := SpeechSpans(probabilities, 55*512, opts)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, []Span{{0, 30*512 + 480}}) {
		t.Fatal("hysteresis/merge", got)
	}
	// Strong speech resuming before minimum silence clears the pending end.
	p := []float32{0.9, 0.9, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9}
	got, err = SpeechSpans(p, int64(len(p)*512), opts)
	if err != nil || !reflect.DeepEqual(got, []Span{{0, int64(len(p) * 512)}}) {
		t.Fatal("short gap", got, err)
	}
}
func TestSpeechSpansRejectMalformed(t *testing.T) {
	base := DefaultSegmentOptions()
	p := make([]float32, 20)
	for _, o := range []SegmentOptions{{}, {Threshold: float32(math.NaN()), MinSilenceSamples: 1}, {Threshold: 1, MinSilenceSamples: 1}, {Threshold: 0.5, MinSpeechSamples: -1, MinSilenceSamples: 1}, {Threshold: 0.5, MinSilenceSamples: 16000 * 61}, {Threshold: 0.5, MinSilenceSamples: 1, PaddingSamples: -1}, {Threshold: 0.5, MinSilenceSamples: 1, PaddingSamples: 16000 * 61}} {
		if _, err := SpeechSpans(p, 20*512, o); err == nil {
			t.Fatal("bad options")
		}
	}
	for _, total := range []int64{0, 4*3600*16000 + 1, 19 * 512} {
		if _, err := SpeechSpans(p, total, base); err == nil {
			t.Fatal("bad total/count")
		}
	}
	for _, v := range []float32{-0.1, 1.1, float32(math.NaN()), float32(math.Inf(1))} {
		p[0] = v
		if _, err := SpeechSpans(p, 20*512, base); err == nil {
			t.Fatal("bad probability")
		}
	}
}
func TestTimeMapBoundaryDirectionAndOwnership(t *testing.T) {
	spans := []Span{{100, 200}, {400, 500}, {900, 950}}
	m, err := NewTimeMap(spans, 1000)
	if err != nil {
		t.Fatal(err)
	}
	spans[0].Start = 0
	if m.Samples() != 250 {
		t.Fatal("speech length")
	}
	for _, tc := range []struct{ sample, start, end int64 }{{0, 100, 100}, {50, 150, 150}, {100, 400, 200}, {101, 401, 401}, {200, 900, 500}, {250, 950, 950}} {
		a, e := m.MapStart(tc.sample)
		if e != nil || a != tc.start {
			t.Fatal("start", tc, a, e)
		}
		b, e := m.MapEnd(tc.sample)
		if e != nil || b != tc.end {
			t.Fatal("end", tc, b, e)
		}
	}
	for _, sample := range []int64{-1, 251} {
		if _, err := m.MapStart(sample); err == nil {
			t.Fatal("out ofrange")
		}
	}
	if (*TimeMap)(nil).Samples() != 0 {
		t.Fatal("nil samples")
	}
	if _, err := (*TimeMap)(nil).MapEnd(0); err == nil {
		t.Fatal("nil mapping")
	}
	empty, err := NewTimeMap(nil, 1000)
	if err != nil || empty.Samples() != 0 {
		t.Fatal(err)
	}
	if _, err := empty.MapStart(0); err == nil {
		t.Fatal("empty mapping")
	}
	for _, spans := range [][]Span{{{-1, 10}}, {{2, 2}}, {{900, 1001}}, {{100, 200}, {150, 300}}, {{400, 500}, {100, 200}}} {
		if _, err := NewTimeMap(spans, 1000); err == nil {
			t.Fatal("invalid spans")
		}
	}
	for _, total := range []int64{0, 4*3600*16000 + 1} {
		if _, err := NewTimeMap(nil, total); err == nil {
			t.Fatal("invalid timeline")
		}
	}
}

func TestSpeechSpansStrictMinimumAndPaddedTailDecision(t *testing.T) {
	opts := DefaultSegmentOptions()
	opts.MinSpeechSamples = 1024
	opts.PaddingSamples = 0
	opts.MinSilenceSamples = 512
	// Exactly the minimum before a confirmed silence is not admitted upstream.
	got, err := SpeechSpans([]float32{.9, .9, 0, 0, 0}, 5*512, opts)
	if err != nil || len(got) != 0 {
		t.Fatal("strict minimum", got, err)
	}
	// Upstream evaluates the tail against padded frame count. Retained PCM is
	// clipped, but its admission decision must not change for a partial EOF.
	got, err = SpeechSpans([]float32{.9, .9, .9}, 1025, opts)
	if err != nil || !reflect.DeepEqual(got, []Span{{Start: 0, End: 1025}}) {
		t.Fatal("padded tail decision", got, err)
	}
	opts.MinSpeechSamples = 1200
	got, err = SpeechSpans([]float32{.9, .9, .9}, 1025, opts)
	if err != nil || !reflect.DeepEqual(got, []Span{{Start: 0, End: 1025}}) {
		t.Fatal("padded decision actual clip", got, err)
	}
}
