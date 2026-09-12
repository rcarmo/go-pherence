package whisper

import (
	"context"
	"errors"
	"math"
	"reflect"
	"testing"
)

// Deterministic fault injection: cancel the real child context when execution
// reaches the Nth checkpoint. This is not a wall-clock latency measurement.
type checkpointContext struct {
	context.Context
	cancel    context.CancelFunc
	at, calls int
}

func newCheckpointContext(at int) *checkpointContext {
	ctx, cancel := context.WithCancel(context.Background())
	return &checkpointContext{Context: ctx, cancel: cancel, at: at}
}
func (c *checkpointContext) Err() error {
	c.calls++
	if c.at > 0 && c.calls == c.at {
		c.cancel()
	}
	return c.Context.Err()
}

func contextToyModel(t *testing.T) *Whisper {
	t.Helper()
	cfg := checkedLoadConfig()
	cfg.MaxLength = 8
	cfg.EncoderLayers = 2
	cfg.DecoderLayers = 2
	source := checkedLoadFixture(cfg, "F32")
	for name, tensor := range source.tensors {
		for i := range tensor.data {
			tensor.data[i] = float32((i*17+len(name))%23-11) / 64
		}
		source.tensors[name] = tensor
	}
	w, err := LoadModelSourceChecked(context.Background(), source, cfg)
	if err != nil {
		t.Fatal(err)
	}
	return w
}

func contextToyMel(w *Whisper) []float32 {
	x := make([]float32, w.Config.NumMelBins*w.Config.MaxLength)
	for i := range x {
		x[i] = float32((i*11)%31-15) / 32
	}
	return x
}
func equalContextFloats(t *testing.T, want, got []float32) {
	t.Helper()
	if len(want) != len(got) {
		t.Fatal("changed shape")
	}
	for i := range want {
		if math.Float32bits(want[i]) != math.Float32bits(got[i]) {
			t.Fatalf("changed arithmetic at %d: %g/%g", i, want[i], got[i])
		}
	}
}

func TestSpeechContextEncoderLegacyParityAndObservers(t *testing.T) {
	w := contextToyModel(t)
	mel := contextToyMel(w)
	type snapshot struct {
		boundary          EncoderBoundary
		layer, rows, cols int
		values            []float32
	}
	collect := func(dst *[]snapshot) EncoderObserver {
		return func(b EncoderBoundary, l, r, c int, x []float32) {
			*dst = append(*dst, snapshot{b, l, r, c, append([]float32(nil), x...)})
		}
	}
	var old, new []snapshot
	want := w.Encoder.ForwardObserved(mel, w.Config.MaxLength, collect(&old))
	got, err := w.Encoder.forwardObservedContext(context.Background(), mel, w.Config.MaxLength, collect(&new))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(old, new) {
		t.Fatal("observer values/order changed")
	}
	equalContextFloats(t, want, got)
	// Public path yields identical values without the observer mechanism.
	got, err = w.Encoder.ForwardContext(context.Background(), mel, w.Config.MaxLength)
	if err != nil {
		t.Fatal(err)
	}
	equalContextFloats(t, want, got)
	x := make([]float32, 4*w.Config.EncoderDModel)
	for i := range x {
		x[i] = float32(i) / 16
	}
	want = w.Encoder.forwardLayer(0, &w.Encoder.Layers[0], x, 4)
	got, err = w.Encoder.forwardLayerContext(context.Background(), 0, &w.Encoder.Layers[0], x, 4)
	if err != nil {
		t.Fatal(err)
	}
	equalContextFloats(t, want, got)
}

func TestSpeechContextEncoderEveryCheckpoint(t *testing.T) {
	w := contextToyModel(t)
	mel := contextToyMel(w)
	count := newCheckpointContext(0)
	defer count.cancel()
	if _, err := w.Encoder.ForwardContext(count, mel, w.Config.MaxLength); err != nil {
		t.Fatal(err)
	}
	t.Logf("encoder cancellation checkpoints exercised: %d", count.calls)
	if count.calls < 35 {
		t.Fatalf("missing per-operator checks: %d", count.calls)
	}
	for at := 1; at <= count.calls; at++ {
		ctx := newCheckpointContext(at)
		output, err := w.Encoder.ForwardContext(ctx, mel, w.Config.MaxLength)
		ctx.cancel()
		if !errors.Is(err, context.Canceled) || output != nil {
			t.Fatalf("checkpoint%d returned partial/success: %v", at, err)
		}
	}
	// A real cancel at an existing graph observer must prevent later operators.
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	original := w.Encoder.Conv2Weight
	w.Encoder.Conv2Weight = nil
	defer func() { w.Encoder.Conv2Weight = original }()
	calls := 0
	got, err := w.Encoder.forwardObservedContext(ctx, mel, w.Config.MaxLength, func(b EncoderBoundary, _, _, _ int, _ []float32) {
		calls++
		if b != EncoderBoundaryConv1 {
			t.Fatal("later observer called")
		}
		cancel()
	})
	if !errors.Is(err, context.Canceled) || got != nil || calls != 1 {
		t.Fatal("cancel after conv1")
	}
}

func TestSpeechContextDecoderStateParityAndCancellation(t *testing.T) {
	w := contextToyModel(t)
	enc := w.Encoder.Forward(contextToyMel(w), w.Config.MaxLength)
	n := len(enc) / w.Config.EncoderDModel
	want := NewDecoderState(w.Config, enc, n, w.Decoder)
	got, err := NewDecoderStateContext(context.Background(), w.Config, enc, n, w.Decoder)
	if err != nil {
		t.Fatal(err)
	}
	for i := range want.CrossK {
		equalContextFloats(t, want.CrossK[i], got.CrossK[i])
		equalContextFloats(t, want.CrossV[i], got.CrossV[i])
		equalContextFloats(t, want.CrossKHead[i], got.CrossKHead[i])
		equalContextFloats(t, want.CrossVHead[i], got.CrossVHead[i])
		if len(got.SelfKCache[i]) != 0 || cap(got.SelfKCache[i]) != cap(want.SelfKCache[i]) {
			t.Fatal("changed self KV capacity")
		}
	}
	if got.Pos != 0 || got.LastToken != -1 {
		t.Fatal("changed initial decoder position")
	}
	// Actual first decoder step verifies that state layout feeds the same code.
	equalContextFloats(t, w.Decoder.ForwardToken(TokenSOT, want), w.Decoder.ForwardToken(TokenSOT, got))
	count := newCheckpointContext(0)
	defer count.cancel()
	if _, err := NewDecoderStateContext(count, w.Config, enc, n, w.Decoder); err != nil {
		t.Fatal(err)
	}
	t.Logf("cross-KV cancellation checkpoints exercised: %d", count.calls)
	if count.calls < 10 {
		t.Fatal("missing projection checks")
	}
	for at := 1; at <= count.calls; at++ {
		ctx := newCheckpointContext(at)
		state, err := NewDecoderStateContext(ctx, w.Config, enc, n, w.Decoder)
		ctx.cancel()
		if !errors.Is(err, context.Canceled) || state != nil {
			t.Fatalf("state checkpoint%d: %v", at, err)
		}
	}
}

func TestSpeechContextPrecancelBeforeTouchingModels(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	var enc *Encoder
	if got, err := enc.ForwardContext(ctx, nil, 0); got != nil || !errors.Is(err, context.Canceled) {
		t.Fatal("encoder precancel")
	}
	if got, err := NewDecoderStateContext(ctx, Config{}, nil, 0, nil); got != nil || !errors.Is(err, context.Canceled) {
		t.Fatal("state precancel")
	}
	if mel, frames, err := MelFlatFromSamplesCheckedContext(ctx, nil, Config{}); mel != nil || frames != 0 || !errors.Is(err, context.Canceled) {
		t.Fatal("frontend precancel")
	}
}

func TestSpeechContextPCMEveryCheckpointNoFailedWindowEmission(t *testing.T) {
	w := contextToyModel(t)
	tok, raw := generationJSONFixture(t, w.Config)
	policy, err := ParseGenerationConfigChecked(marshalConfig(t, raw), w.Config, tok)
	if err != nil {
		t.Fatal(err)
	}
	opts := PCMTranscribeOptions{Language: "pt", Generation: policy}
	reader := sampleReadFunc(func(_ context.Context, dst []float32, _ int64) (int, error) {
		for i := range dst {
			dst[i] = float32(i%5) / 8
		}
		return len(dst), nil
	})
	count := newCheckpointContext(0)
	defer count.cancel()
	// The artificial model need not produce useful text. Count until either its
	// expected strict decode failure or successful completion; not a quality gate.
	_ = w.TranscribePCMWindows(count, reader, 1280, tok, opts, func(WindowTranscript) error { return nil })
	t.Logf("PCM cancellation checkpoints exercised: %d", count.calls)
	if count.calls < 50 {
		t.Fatalf("missing stage checkpoints:%d", count.calls)
	}
	// Check representative positions across the complete invocation, including
	// validation, frontend, encoder, cross-KV and token selection. No sleep/race.
	for at := 1; at <= count.calls; at++ {
		ctx := newCheckpointContext(at)
		emitted := 0
		err := w.TranscribePCMWindows(ctx, reader, 1280, tok, opts, func(WindowTranscript) error { emitted++; return nil })
		ctx.cancel()
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("PCM checkpoint%d/%d: %v", at, count.calls, err)
		}
		// A callback completed before cancellation is valid. Never more than one
		// window is emitted; there is no asynchronous tail or failed-window callback.
		if emitted > 1 {
			t.Fatal("repeated callback")
		}
		if err := w.TranscribePCMWindows(context.Background(), reader, 0, tok, opts, func(WindowTranscript) error { t.Fatal("empty emitted"); return nil }); err != nil {
			t.Fatal("gate not released", err)
		}
	}
}
