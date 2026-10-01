package whisper

import (
	"context"
	"errors"
	"math"
	"os"
	"testing"
	"time"
)

func TestDecoderParallelRowsTrainedExact(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_DECODER_ROWS_TRAINED") != "1" {
		t.Skip("explicit pinned decoder rows qualification")
	}
	d, ok := t.Deadline()
	if !ok || time.Until(d) > 120*time.Second {
		t.Fatal("bounded120s")
	}
	ctx, stop := context.WithDeadline(context.Background(), d.Add(-time.Second))
	defer stop()
	t.Setenv("GO_PHERENCE_DISABLE_NVIDIA", "1")
	model, _, _, file := pinnedLegacyWhisperModelMode(t, ctx, true)
	defer file.Close()
	rows := model.Config.MaxLength / 2
	encoded := make([]float32, rows*model.Config.DecoderDModel)
	for i := range encoded {
		encoded[i] = float32(i%71-35) * .0027
	}
	a, e := NewDecoderStateContext(ctx, model.Config, encoded, rows, model.Decoder)
	if e != nil {
		t.Fatal(e)
	}
	b, e := NewDecoderStateContext(ctx, model.Config, encoded, rows, model.Decoder)
	if e != nil {
		t.Fatal(e)
	}
	check := func(a, b []float32) {
		t.Helper()
		if len(a) != len(b) {
			t.Fatal("extent")
		}
		for i, v := range a {
			if math.Float32bits(v) != math.Float32bits(b[i]) {
				t.Fatal("bits", i, v, b[i])
			}
		}
	}
	var oa, ob []float32
	a.CrossAttentionObserver = func(_, _, _ int, v []float32) { oa = append(oa, v...) }
	b.CrossAttentionObserver = func(_, _, _ int, v []float32) { ob = append(ob, v...) }
	for _, token := range []int{50258, 50259, 50359, 50364, 400, 370, 1029, 406} {
		t.Setenv(envDecoderParallelRows, "0")
		la := append([]float32(nil), model.Decoder.ForwardToken(token, a)...)
		t.Setenv(envDecoderParallelRows, "1")
		lb := model.Decoder.ForwardToken(token, b)
		if decoderRowWorkers(model.Config.DecoderFFNDim) != 4 {
			t.Fatal("4workers")
		}
		check(la, lb)
		for l := 0; l < model.Config.DecoderLayers; l++ {
			check(a.SelfKCache[l], b.SelfKCache[l])
			check(a.SelfVCache[l], b.SelfVCache[l])
		}
	}
	check(oa, ob)
	for l := 0; l < model.Config.DecoderLayers; l++ {
		check(a.CrossK[l], b.CrossK[l])
		check(a.CrossV[l], b.CrossV[l])
	}
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	if s, e := NewDecoderStateContext(cancelled, model.Config, encoded, rows, model.Decoder); s != nil || !errors.Is(e, context.Canceled) {
		t.Fatal("precancel", e)
	}
	fresh, e := NewDecoderStateContext(ctx, model.Config, encoded, rows, model.Decoder)
	if e != nil {
		t.Fatal(e)
	}
	ref, e := NewDecoderStateContext(ctx, model.Config, encoded, rows, model.Decoder)
	if e != nil {
		t.Fatal(e)
	}
	t.Setenv(envDecoderParallelRows, "0")
	la := append([]float32(nil), model.Decoder.ForwardToken(50258, ref)...)
	t.Setenv(envDecoderParallelRows, "1")
	check(la, model.Decoder.ForwardToken(50258, fresh))
	t.Logf("DECODER_PARALLEL_LOGITS_SELFKV_CROSS_OBSERVER_EXACT tokens=8 vocab=%d observerValues=%d precancel_freshreuse=PASS", model.Config.VocabSize, len(oa))
}
