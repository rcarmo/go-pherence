package whisper

import (
	"context"
	"errors"
	"math"
	"testing"
)

func TestPCMDigitalSilence(t *testing.T) {
	ctx := context.Background()
	for _, v := range []float32{0, float32(math.Copysign(0, -1)), math.SmallestNonzeroFloat32, -math.SmallestNonzeroFloat32, 1e-12, float32(math.NaN()), float32(math.Inf(1))} {
		got, err := pcmDigitalSilence(ctx, []float32{0, v, 0})
		if err != nil || got != (v == 0) {
			t.Fatal(v, got, err)
		}
	}
	for _, at := range []int{1, 2, 3, 4} {
		c := newCheckpointContext(at)
		_, err := pcmDigitalSilence(c, make([]float32, 40000))
		c.cancel()
		if !errors.Is(err, context.Canceled) {
			t.Fatal("silencescan cancellation", at, err)
		}
	}
}
func TestPCMDigitalSilenceNoInference(t *testing.T) {
	w := toyPCMModel()
	w.Encoder.Conv1Weight[0] = float32(math.NaN())
	tok := checkedTestTokenizer(w.Config.VocabSize)
	calls := 0
	reader := sampleReadFunc(func(_ context.Context, dst []float32, _ int64) (int, error) { clear(dst); return len(dst), nil })
	opts := PCMTranscribeOptions{Language: "en", SkipDigitalSilence: true}
	if err := w.TranscribePCMWindows(context.Background(), reader, 321, tok, opts, func(out WindowTranscript) error {
		calls++
		if len(out.Segments) != 0 {
			t.Fatal("silence text")
		}
		return nil
	}); err != nil || calls != 2 {
		t.Fatal(err, calls)
	}
	// Defaults unchanged; poisoned model must run/error without shortcut.
	opts.SkipDigitalSilence = false
	if err := w.TranscribePCMWindows(context.Background(), reader, 321, tok, opts, func(WindowTranscript) error { return nil }); err == nil {
		t.Fatal("default unexpectedlyskipped")
	}
	opts.SkipDigitalSilence = true
	for _, value := range []float32{math.SmallestNonzeroFloat32, float32(math.NaN()), float32(math.Inf(1))} {
		reader := sampleReadFunc(func(_ context.Context, dst []float32, _ int64) (int, error) {
			clear(dst)
			dst[0] = value
			return len(dst), nil
		})
		if err := w.TranscribePCMWindows(context.Background(), reader, 1, tok, opts, func(WindowTranscript) error { return nil }); err == nil {
			t.Fatal("nonzero/nonfinite dropped", value)
		}
	}
	// Missing weights still fail preflight, even for digital silence.
	w.Decoder.TokenEmbed = nil
	if err := w.TranscribePCMWindows(context.Background(), reader, 1, tok, opts, func(WindowTranscript) error { return nil }); err == nil {
		t.Fatal("skippedmodelvalidation")
	}
}
