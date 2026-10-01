package whisper

import (
	"context"
	"errors"
	"math"
	"reflect"
	"testing"
)

func TestAlignmentDecoderStateSharesOnlyImmutableCrossKV(t *testing.T) {
	t.Setenv("GO_PHERENCE_DISABLE_NVIDIA", "1")
	w := contextToyModel(t)
	ctx := context.Background()
	encoded, err := w.Encoder.ForwardContext(ctx, make([]float32, w.Config.MaxLength*w.Config.NumMelBins), w.Config.MaxLength)
	if err != nil {
		t.Fatal(err)
	}
	n := (w.Config.MaxLength + 1) / 2
	source, err := NewDecoderStateContext(ctx, w.Config, encoded, n, w.Decoder)
	if err != nil {
		t.Fatal(err)
	}
	w.Decoder.ForwardToken(1, source)
	copied, err := NewDecoderStateContext(ctx, w.Config, encoded, n, w.Decoder)
	if err != nil {
		t.Fatal(err)
	}
	shared, err := newAlignmentDecoderStateContext(ctx, w.Config, n, source)
	if err != nil {
		t.Fatal(err)
	}
	if shared.Pos != 0 || shared.LastToken != -1 || shared.Bufs == source.Bufs || shared.CrossAttentionObserver != nil {
		t.Fatal("fresh state")
	}
	snapshots := make([][]float32, 0)
	for l := 0; l < w.Config.DecoderLayers; l++ {
		if &source.CrossK[l][0] != &shared.CrossK[l][0] || &source.CrossV[l][0] != &shared.CrossV[l][0] || &source.CrossKHead[l][0] != &shared.CrossKHead[l][0] || &source.CrossVHead[l][0] != &shared.CrossVHead[l][0] {
			t.Fatal("cross storage copied")
		}
		snapshots = append(snapshots, append([]float32(nil), source.CrossK[l]...), append([]float32(nil), source.CrossV[l]...), append([]float32(nil), source.CrossKHead[l]...), append([]float32(nil), source.CrossVHead[l]...))
		if len(shared.SelfKCache[l]) != 0 || len(shared.SelfVCache[l]) != 0 || &shared.SelfKCache[l][:cap(shared.SelfKCache[l])][0] == &source.SelfKCache[l][:cap(source.SelfKCache[l])][0] {
			t.Fatal("self storage shared")
		}
	}
	var observedA, observedB []float32
	copied.CrossAttentionObserver = func(_, _, _ int, v []float32) { observedA = append(observedA, v...) }
	shared.CrossAttentionObserver = func(_, _, _ int, v []float32) { observedB = append(observedB, v...) }
	for _, token := range []int{1, 2, 3} {
		a := append([]float32(nil), w.Decoder.ForwardToken(token, copied)...)
		b := w.Decoder.ForwardToken(token, shared)
		equalContextFloats(t, a, b)
	}
	equalContextFloats(t, observedA, observedB)
	for l := 0; l < w.Config.DecoderLayers; l++ {
		for j, v := range [][]float32{source.CrossK[l], source.CrossV[l], source.CrossKHead[l], source.CrossVHead[l]} {
			if !reflect.DeepEqual(v, snapshots[l*4+j]) {
				t.Fatal("cross mutated")
			}
		}
	}
	shared.CrossK[0] = nil
	if len(source.CrossK[0]) == 0 {
		t.Fatal("outer slices shared")
	}
}

func TestAlignmentDecoderStateCancellationAndGeometry(t *testing.T) {
	t.Setenv("GO_PHERENCE_DISABLE_NVIDIA", "1")
	w := contextToyModel(t)
	ctx := context.Background()
	n := 4
	source, err := NewDecoderStateContext(ctx, w.Config, make([]float32, n*w.Config.DecoderDModel), n, w.Decoder)
	if err != nil {
		t.Fatal(err)
	}
	count := newCheckpointContext(0)
	if _, err := newAlignmentDecoderStateContext(count, w.Config, n, source); err != nil {
		t.Fatal(err)
	}
	count.cancel()
	for at := 1; at <= count.calls; at++ {
		fault := newCheckpointContext(at)
		got, err := newAlignmentDecoderStateContext(fault, w.Config, n, source)
		fault.cancel()
		if got != nil || !errors.Is(err, context.Canceled) {
			t.Fatal("transaction", at, got, err)
		}
	}
	for _, bad := range []*DecoderState{nil, {}, {CrossK: source.CrossK, CrossV: source.CrossV, CrossKHead: source.CrossKHead}} {
		if _, err := newAlignmentDecoderStateContext(ctx, w.Config, n, bad); err == nil {
			t.Fatal("bad geometry")
		}
	}
	if _, err := newAlignmentDecoderStateContext(nil, w.Config, n, source); err == nil {
		t.Fatal("nilctx")
	}
	if _, err := newAlignmentDecoderStateContext(ctx, w.Config, n+1, source); err == nil {
		t.Fatal("mismatched extent")
	}
}

func TestAlignmentDecoderStateAllocationReduction(t *testing.T) {
	t.Setenv("GO_PHERENCE_DISABLE_NVIDIA", "1")
	w := contextToyModel(t)
	ctx := context.Background()
	n := 4
	x := make([]float32, n*w.Config.DecoderDModel)
	source, err := NewDecoderStateContext(ctx, w.Config, x, n, w.Decoder)
	if err != nil {
		t.Fatal(err)
	}
	old := testing.AllocsPerRun(10, func() {
		s, e := NewDecoderStateContext(ctx, w.Config, x, n, w.Decoder)
		if e != nil || s == nil {
			panic(e)
		}
	})
	shared := testing.AllocsPerRun(10, func() {
		s, e := newAlignmentDecoderStateContext(ctx, w.Config, n, source)
		if e != nil || s == nil {
			panic(e)
		}
	})
	if shared >= old || math.IsNaN(shared) {
		t.Fatal("no allocation benefit", old, shared)
	}
	t.Log("allocations", old, "->", shared)
}
