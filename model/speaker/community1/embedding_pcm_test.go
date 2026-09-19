package community1

import (
	"context"
	"errors"
	"github.com/rcarmo/go-pherence/loader/audio"
	"math"
	"reflect"
	"sync"
	"testing"
)

func embeddingPCMFixture(t *testing.T) (*ExperimentalEmbedding, []float32) {
	t.Helper()
	c := loadResNetFixtures(t)[2]
	m, e := NewWeSpeakerResNet34(context.Background(), c.Config, c.Weights)
	if e != nil {
		t.Fatal(e)
	}
	w, e := NewExperimentalEmbedding(context.Background(), m)
	if e != nil {
		t.Fatal(e)
	}
	pcm := make([]float32, 2960)
	for i := range pcm {
		pcm[i] = float32(math.Sin(float64(i)*.071)) * .13
	}
	return w, pcm
}
func TestExperimentalEmbeddingCompositionAndOwner(t *testing.T) {
	m, pcm := embeddingPCMFixture(t)
	ctx := context.Background()
	before := append([]float32(nil), pcm...)
	f, n, e := audio.WeSpeakerFbank(ctx, pcm)
	if e != nil {
		t.Fatal(e)
	}
	for _, mode := range []WeSpeakerBlockMode{WeSpeakerBlockScalar, WeSpeakerBlockSIMD, WeSpeakerBlockGEMM} {
		calls := 0
		owned, e := m.ForwardPCMFramesObserved(ctx, pcm, mode, EmbeddingPCMObservers{Trunk: func(_, _ int, _ CHWShape, _ []float32) { calls++ }})
		if e != nil {
			t.Fatal(e)
		}
		want, s, e := m.model.ForwardFrames(ctx, f, n, mode)
		if e != nil {
			t.Fatal(e)
		}
		shape, samples, frames := owned.Shape()
		if calls != 17 || samples != len(pcm) || frames != n || shape != s || !reflect.DeepEqual(owned.values, want) || !reflect.DeepEqual(before, pcm) {
			t.Fatal("trunk composition")
		}
		for _, mask := range [][]float32{nil, {1, .3, 0, .8, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0}} {
			speakers, maskFrames := 0, 0
			if mask != nil {
				speakers, maskFrames = 3, 5
			}
			got, e := m.EmbedFrames(ctx, owned, mask, speakers, maskFrames, mode)
			if e != nil {
				t.Fatal(e)
			}
			expected, e := m.model.ForwardEmbedding(ctx, want, s, mask, speakers, maskFrames, mode)
			if e != nil || !reflect.DeepEqual(got, expected) {
				t.Fatal("pool composition", e)
			}
			got.Embeddings[0] = 999
			again, e := m.EmbedFrames(ctx, owned, mask, speakers, maskFrames, mode)
			if e != nil || !reflect.DeepEqual(again, expected) {
				t.Fatal("result alias")
			}
		}
		other, e := NewExperimentalEmbedding(ctx, m.model)
		if e != nil {
			t.Fatal(e)
		}
		if r, e := other.EmbedFrames(ctx, owned, nil, 0, 0, mode); r != nil || e == nil {
			t.Fatal("cross-owner reuse")
		}
	}
}
func TestExperimentalEmbeddingRejectsAndCancels(t *testing.T) {
	testExperimentalEmbeddingRejectsAndCancels(t, WeSpeakerBlockSIMD)
}
func TestExperimentalEmbeddingGEMMRejectsAndCancels(t *testing.T) {
	testExperimentalEmbeddingRejectsAndCancels(t, WeSpeakerBlockGEMM)
}
func testExperimentalEmbeddingRejectsAndCancels(t *testing.T, mode WeSpeakerBlockMode) {
	m, pcm := embeddingPCMFixture(t)
	ctx := context.Background()
	var empty *ExperimentalEmbedding
	for _, bad := range []*ExperimentalEmbedding{nil, {}} {
		if r, e := bad.ForwardPCMFrames(ctx, pcm, mode); r != nil || e == nil {
			t.Fatal("nil model")
		}
	}
	if r, e := NewExperimentalEmbedding(ctx, nil); r != nil || e == nil {
		t.Fatal("nil constructor")
	}
	c := loadResNetFixtures(t)[0]
	wrong, e := NewWeSpeakerResNet34(ctx, c.Config, c.Weights)
	if e != nil {
		t.Fatal(e)
	}
	if r, e := NewExperimentalEmbedding(ctx, wrong); r != nil || e == nil {
		t.Fatal("non80 model")
	}
	for _, bad := range [][]float32{nil, make([]float32, 399), make([]float32, 160001), append([]float32{float32(math.NaN())}, pcm...)} {
		if r, e := m.ForwardPCMFrames(ctx, bad, mode); r != nil || e == nil {
			t.Fatal("bad PCM")
		}
	}
	called := false
	if r, e := m.ForwardPCMFramesObserved(ctx, pcm, 99, EmbeddingPCMObservers{Fbank: func(string, int, []float32) { called = true }}); r != nil || e == nil || called {
		t.Fatal("mode reached frontend")
	}
	cc, cancel := context.WithCancel(ctx)
	cancel()
	if r, e := empty.ForwardPCMFrames(cc, nil, mode); r != nil || !errors.Is(e, context.Canceled) {
		t.Fatal("precancel")
	}
	for target := 0; target < 17; target++ {
		cc, cancel := context.WithCancel(ctx)
		calls := 0
		r, e := m.ForwardPCMFramesObserved(cc, pcm, mode, EmbeddingPCMObservers{Trunk: func(_, _ int, _ CHWShape, _ []float32) {
			if calls == target {
				cancel()
			}
			calls++
		}})
		cancel()
		if r != nil || !errors.Is(e, context.Canceled) || calls != target+1 {
			t.Fatal("trunk cancel", target, e)
		}
	}
	for _, stage := range []string{"window", "power", "log_mel", "centered"} {
		cc, cancel := context.WithCancel(ctx)
		r, e := m.ForwardPCMFramesObserved(cc, pcm, mode, EmbeddingPCMObservers{Fbank: func(s string, _ int, _ []float32) {
			if s == stage {
				cancel()
			}
		}})
		cancel()
		if r != nil || !errors.Is(e, context.Canceled) {
			t.Fatal(stage, e)
		}
	}
	owned, e := m.ForwardPCMFrames(ctx, pcm, mode)
	if e != nil {
		t.Fatal(e)
	}
	called = false
	if r, e := m.EmbedFramesObserved(ctx, owned, nil, 0, 0, 99, func(string, int, int, []float32) { called = true }); r != nil || e == nil || called {
		t.Fatal("invalid pooling mode reached observer")
	}
	for _, stage := range []string{"stats", "embedding"} {
		cc, cancel := context.WithCancel(ctx)
		r, e := m.EmbedFramesObserved(cc, owned, nil, 0, 0, mode, func(s string, _, _ int, _ []float32) {
			if s == stage {
				cancel()
			}
		})
		cancel()
		if r != nil || !errors.Is(e, context.Canceled) {
			t.Fatal(stage, e)
		}
	}
	for _, bad := range []*EmbeddingPCMFrames{nil, {}, {owner: m}} {
		if r, e := m.EmbedFrames(ctx, bad, nil, 0, 0, mode); r != nil || e == nil {
			t.Fatal("bad frames")
		}
	}
	for _, mask := range [][]float32{{-1}, {float32(math.NaN())}} {
		if r, e := m.EmbedFrames(ctx, owned, mask, 1, 1, mode); r != nil || e == nil {
			t.Fatal("bad mask")
		}
	}
	count := newPowersetContext(0)
	_, e = m.ForwardPCMFrames(count, pcm, mode)
	count.cancel()
	if e != nil {
		t.Fatal(e)
	}
	for at := 1; at <= count.calls; at += max(1, count.calls/25) {
		cc := newPowersetContext(at)
		r, e := m.ForwardPCMFrames(cc, pcm, mode)
		cc.cancel()
		if r != nil || !errors.Is(e, context.Canceled) {
			t.Fatal("internal cancel", at, e)
		}
	}
}
func TestExperimentalEmbeddingSharedFramesConcurrency(t *testing.T) {
	m, pcm := embeddingPCMFixture(t)
	ctx := context.Background()
	frames, e := m.ForwardPCMFrames(ctx, pcm, WeSpeakerBlockSIMD)
	if e != nil {
		t.Fatal(e)
	}
	want, e := m.EmbedFrames(ctx, frames, nil, 0, 0, WeSpeakerBlockSIMD)
	if e != nil {
		t.Fatal(e)
	}
	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			got, e := m.EmbedFrames(ctx, frames, nil, 0, 0, WeSpeakerBlockSIMD)
			if e != nil || !reflect.DeepEqual(got, want) {
				t.Error("shared frame mutation", e)
			}
		}()
	}
	wg.Wait()
}
