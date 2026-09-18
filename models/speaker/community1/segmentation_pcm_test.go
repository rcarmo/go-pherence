package community1

import (
	"context"
	"errors"
	"math"
	"reflect"
	"sync"
	"testing"
)

func experimentalFixture(t *testing.T) (*ExperimentalSegmentation, []float32) {
	t.Helper()
	c := segmentationLoadOracles(t)[0]
	m, e := LoadSegmentationSource(context.Background(), segmentationFixtureSource(c), c.Config)
	if e != nil {
		t.Fatal(e)
	}
	filters, e := sincNetFilters(context.Background(), m.sincnet.LowHz, m.sincnet.BandHz)
	if e != nil {
		t.Fatal(e)
	}
	model, e := NewExperimentalSegmentation(context.Background(), m, filters)
	if e != nil {
		t.Fatal(e)
	}
	pcm := make([]float32, 1600)
	for i := range pcm {
		pcm[i] = float32(math.Sin(float64(i)*.071)) * .13
	}
	return model, pcm
}
func TestExperimentalSegmentationComposition(t *testing.T) {
	m, pcm := experimentalFixture(t)
	ctx := context.Background()
	before := append([]float32(nil), pcm...)
	for _, modes := range []SegmentationModes{{SincNetScalarFMA, LSTMScalar, HeadScalar}, {SincNetSIMDFMA, LSTMSIMD, HeadSIMD}} {
		f, grid, e := m.frontend.Forward(ctx, pcm, modes.SincNet)
		if e != nil {
			t.Fatal(e)
		}
		want, e := m.checkpoint.ForwardFeatures(ctx, f, grid.Frames, modes.LSTM, modes.Head)
		if e != nil {
			t.Fatal(e)
		}
		calls := []string{}
		got, e := m.ForwardPCMObserved(ctx, pcm, modes, SegmentationObservers{SincNet: func(_, _, _ int, _ []float32) { calls = append(calls, "sinc") }, LSTM: func(_, _, _ int, _ []float32) { calls = append(calls, "lstm") }, Head: func(_, _, _ int, _ []float32) { calls = append(calls, "head") }})
		if e != nil {
			t.Fatal(e)
		}
		if got.Grid != grid || got.Classes != 7 || !reflect.DeepEqual(got.LogProbabilities, want) || !reflect.DeepEqual(pcm, before) {
			t.Fatal("composition/ownership")
		}
		if !reflect.DeepEqual(calls, []string{"sinc", "sinc", "sinc", "sinc", "lstm", "lstm", "head", "head", "head"}) {
			t.Fatal("observer order", calls)
		}
		got.LogProbabilities[0] = 999
		again, e := m.ForwardPCM(ctx, pcm, modes)
		if e != nil || !reflect.DeepEqual(again.LogProbabilities, want) {
			t.Fatal("retained result alias", e)
		}
	}
}
func TestExperimentalSegmentationRejectsAndCancels(t *testing.T) {
	m, pcm := experimentalFixture(t)
	ctx := context.Background()
	mode := SegmentationModes{SincNetSIMDFMA, LSTMSIMD, HeadSIMD}
	for _, bad := range []SegmentationModes{{}, {SincNetSIMD, LSTMSIMD, HeadSIMD}, {SincNetSIMDFMA, 99, HeadSIMD}, {SincNetSIMDFMA, LSTMSIMD, 99}} {
		called := false
		r, e := m.ForwardPCMObserved(ctx, pcm, bad, SegmentationObservers{SincNet: func(_, _, _ int, _ []float32) { called = true }})
		if r != nil || e == nil || called {
			t.Fatal("bad mode reached frontend")
		}
	}
	var nilModel *ExperimentalSegmentation
	for _, bad := range []*ExperimentalSegmentation{nilModel, {}, {checkpoint: m.checkpoint}} {
		if r, e := bad.ForwardPCM(ctx, pcm, mode); r != nil || e == nil {
			t.Fatal("nil/zero")
		}
	}
	if model, e := NewExperimentalSegmentation(ctx, nil, nil); model != nil || e == nil {
		t.Fatal("nil checkpoint")
	}
	if model, e := NewExperimentalSegmentation(ctx, m.checkpoint, nil); model != nil || e == nil {
		t.Fatal("bad filters")
	}
	for _, input := range [][]float32{nil, make([]float32, 160001), {float32(math.NaN())}} {
		if r, e := m.ForwardPCM(ctx, input, mode); r != nil || e == nil {
			t.Fatal("invalid PCM")
		}
	}
	cc, cancel := context.WithCancel(ctx)
	cancel()
	if r, e := nilModel.ForwardPCM(cc, nil, mode); r != nil || !errors.Is(e, context.Canceled) {
		t.Fatal("precancel")
	}
	for target := 0; target < 9; target++ {
		cc, cancel := context.WithCancel(ctx)
		calls := 0
		observe := func(_, _, _ int, _ []float32) {
			if calls == target {
				cancel()
			}
			calls++
		}
		r, e := m.ForwardPCMObserved(cc, pcm, mode, SegmentationObservers{observe, observe, observe})
		cancel()
		if r != nil || !errors.Is(e, context.Canceled) || calls != target+1 {
			t.Fatal("boundary cancel", target, e, calls)
		}
	}
	count := newPowersetContext(0)
	_, e := m.ForwardPCM(count, pcm, mode)
	count.cancel()
	if e != nil {
		t.Fatal(e)
	}
	for at := 1; at <= count.calls; at += max(1, count.calls/40) {
		cc := newPowersetContext(at)
		r, e := m.ForwardPCM(cc, pcm, mode)
		cc.cancel()
		if r != nil || !errors.Is(e, context.Canceled) {
			t.Fatal("checkpoint cancel", at, e)
		}
	}
}
func TestExperimentalSegmentationConcurrentCalls(t *testing.T) {
	m, pcm := experimentalFixture(t)
	mode := SegmentationModes{SincNetSIMDFMA, LSTMSIMD, HeadSIMD}
	want, e := m.ForwardPCM(context.Background(), pcm, mode)
	if e != nil {
		t.Fatal(e)
	}
	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			got, e := m.ForwardPCM(context.Background(), pcm, mode)
			if e != nil {
				t.Error(e)
				return
			}
			if !reflect.DeepEqual(got, want) {
				t.Error("shared mutable state")
			}
		}()
	}
	wg.Wait()
}
