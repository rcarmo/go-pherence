package omnivoice

import (
	"math"
	"reflect"
	"testing"
)

func TestDefaultSilenceOptions(t *testing.T) {
	got := DefaultSilenceOptions()
	if got.MidSilenceMS != 300 || got.LeadingKeepMS != 100 || got.TrailingKeepMS != 300 || got.SilenceThresholdDB != -50 || got.MaxSamples != 20*24000 {
		t.Fatalf("unexpected defaults: %+v", got)
	}
	ref := ReferenceSilenceOptions()
	if ref.MidSilenceMS != 200 || ref.LeadingKeepMS != 100 || ref.TrailingKeepMS != 200 || ref.SilenceThresholdDB != -50 || ref.MaxSamples != 20*24000 {
		t.Fatalf("unexpected reference defaults: %+v", ref)
	}
}

func TestPCM16QuantizationParity(t *testing.T) {
	in := []float32{0, 0.5, -0.5, 0.99999, -0.99999, 1, -1, 1.1, -1.1, float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))}
	want := []int16{0, 16384, -16384, 32767, -32767, 32767, -32768, 32767, -32768, 0, 32767, -32768}
	if got := quantizeMonoPCM16(in); !reflect.DeepEqual(got, want) {
		t.Fatalf("quantized=%v want %v", got, want)
	}
	back := dequantizeMonoPCM16(want)
	if back[1] != 0.5 || back[2] != -0.5 || back[5] != 32767.0/32768.0 || back[6] != -1 {
		t.Fatalf("dequantized=%v", back)
	}
}

func TestAudioLenMSMatchesPythonRound(t *testing.T) {
	cases := map[int]int{
		0:  0,
		11: 0,
		12: 0,
		13: 1,
		36: 2,
		60: 2,
		84: 4,
	}
	for frames, want := range cases {
		if got := audioLenMS(frames); got != want {
			t.Fatalf("audioLenMS(%d)=%d want %d", frames, got, want)
		}
	}
}

func TestRemoveSilenceLongMiddleDefault(t *testing.T) {
	audio := append(append(toneMS(100, 0.5), silenceMS(900)...), toneMS(100, -0.5)...)
	got, err := RemoveSilenceMono24k(audio, DefaultSilenceOptions())
	if err != nil {
		t.Fatal(err)
	}
	want := append(append(toneMS(100, 0.5), silenceMS(600)...), toneMS(100, -0.5)...)
	assertFloatSlicesNear(t, got, quantizedFloatCopy(want), 0)
}

func TestRemoveSilenceKeepsShortMiddleGapBoundaries(t *testing.T) {
	audio := append(append(toneMS(100, 0.5), silenceMS(350)...), toneMS(100, 0.25)...)
	got, err := RemoveSilenceMono24k(audio, DefaultSilenceOptions())
	if err != nil {
		t.Fatal(err)
	}
	assertFloatSlicesNear(t, got, quantizedFloatCopy(audio), 0)
}

func TestRemoveSilenceReferencePresetTrimsEdges(t *testing.T) {
	audio := append(append(silenceMS(500), toneMS(100, 0.5)...), silenceMS(800)...)
	got, err := RemoveReferenceSilenceMono24k(audio)
	if err != nil {
		t.Fatal(err)
	}
	want := append(append(silenceMS(100), toneMS(100, 0.5)...), silenceMS(200)...)
	assertFloatSlicesNear(t, got, quantizedFloatCopy(want), 0)
}

func TestRemoveSilenceAllSilentAndLowAmplitude(t *testing.T) {
	cases := [][]float32{
		silenceMS(1000),
		toneMS(1000, 0.001),
	}
	for _, in := range cases {
		got, err := RemoveSilenceMono24k(in, DefaultSilenceOptions())
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != 0 {
			t.Fatalf("expected empty output, got %d samples", len(got))
		}
	}
}

func TestRemoveSilenceRejectsInvalidInput(t *testing.T) {
	tooLong := make([]float32, 20*24000+1)
	if _, err := RemoveSilenceMono24k(tooLong, DefaultSilenceOptions()); err == nil {
		t.Fatal("accepted oversized reference audio")
	}
	if _, err := RemoveSilenceMono24k([]float32{float32(math.NaN())}, DefaultSilenceOptions()); err == nil {
		t.Fatal("accepted NaN")
	}
}

func TestFadeAndPadCustomAndFlags(t *testing.T) {
	in := []float32{1, 2, 3, 4}
	got, err := FadeAndPadMono24k(in, FadePadOptions{PadDuration: 1 / 24000.0, FadeDuration: 2 / 24000.0, FadeIn: true, FadeOut: true, MaxSamples: 100})
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{0, 0, 2, 3, 0, 0}
	assertFloatSlicesNear(t, got, want, 1e-7)
	toneSide, err := FadeAndPadMono24k(in, FadePadOptions{PadDuration: 0, FadeDuration: 2 / 24000.0, FadeIn: false, FadeOut: true, MaxSamples: 100})
	if err != nil {
		t.Fatal(err)
	}
	assertFloatSlicesNear(t, toneSide, []float32{1, 2, 3, 0}, 1e-7)
}

func TestFadeAndPadDefaultsAndValidation(t *testing.T) {
	got, err := FadeAndPadMono24k([]float32{1, 1, 1, 1}, DefaultFadePadOptions())
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 4+2*2400 {
		t.Fatalf("len=%d want %d", len(got), 4+2*2400)
	}
	if got[2400] != 0 || got[2401] != 1 || got[2402] != 1 || got[2403] != 0 {
		t.Fatalf("unexpected faded center %v", got[2400:2404])
	}
	if _, err := FadeAndPadMono24k(make([]float32, 10*24000+1), DefaultFadePadOptions()); err == nil {
		t.Fatal("accepted oversized output audio")
	}
	if _, err := FadeAndPadMono24k([]float32{1}, FadePadOptions{PadDuration: -1, FadeDuration: 0, FadeIn: true, FadeOut: true, MaxSamples: 1}); err == nil {
		t.Fatal("accepted invalid pad duration")
	}
}

func toneMS(ms int, amp float32) []float32 {
	out := make([]float32, msToFrames(ms))
	for i := range out {
		out[i] = amp
	}
	return out
}

func silenceMS(ms int) []float32 {
	return make([]float32, msToFrames(ms))
}

func quantizedFloatCopy(in []float32) []float32 {
	return dequantizeMonoPCM16(quantizeMonoPCM16(in))
}

func assertFloatSlicesNear(t *testing.T, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("len=%d want %d", len(got), len(want))
	}
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > tol {
			t.Fatalf("sample[%d]=%g want %g (tol=%g)", i, got[i], want[i], tol)
		}
	}
}

func TestPostprocessBounds(t *testing.T) {
	for _, n := range []int{-1, 0, int(^uint(0) >> 1)} {
		o := DefaultSilenceOptions()
		o.MaxSamples = n
		if _, err := RemoveSilenceMono24k([]float32{.1}, o); err == nil {
			t.Fatal("bad max accepted")
		}
	}
	o := DefaultSilenceOptions()
	o.MidSilenceMS = int(^uint(0) >> 1)
	if _, err := RemoveSilenceMono24k([]float32{.1}, o); err == nil {
		t.Fatal("huge milliseconds accepted")
	}
	f := DefaultFadePadOptions()
	f.PadDuration = math.MaxFloat64
	if _, err := FadeAndPadMono24k([]float32{.1}, f); err == nil {
		t.Fatal("huge duration accepted")
	}
	f = DefaultFadePadOptions()
	f.PadDuration = 10
	if _, err := FadeAndPadMono24k([]float32{.1}, f); err == nil {
		t.Fatal("oversize output accepted")
	}
	f = DefaultFadePadOptions()
	out, err := FadeAndPadMono24k(make([]float32, 240000), f)
	if err != nil || len(out) != 244800 {
		t.Fatal("10 second input boundary", err, len(out))
	}
}
