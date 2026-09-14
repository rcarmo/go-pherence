package community1

import (
	"bytes"
	"compress/gzip"
	"context"
	"encoding/json"
	"errors"
	"io"
	"math"
	"os"
	"reflect"
	"sync"
	"testing"
)

type resnetMaskCase struct {
	Masks, Statistics, Embeddings []float32
	Speakers                      int
	MaskFrames                    int       `json:"mask_frames"`
	WeightSum                     []float32 `json:"weight_sum"`
	NonzeroFrames                 []int     `json:"nonzero_frames"`
}
type resnetCase struct {
	Config     WeSpeakerResNetConfig
	Frames     int
	Input      []float32
	Weights    WeSpeakerResNetWeights
	Boundaries []struct {
		Stage, Block int
		Shape        CHWShape
		Values       []float32
	}
	MaskCases []resnetMaskCase `json:"mask_cases"`
}

func loadResNetFixtures(t *testing.T) []resnetCase {
	t.Helper()
	data, err := os.ReadFile("testdata/resnet-reference.json.gz")
	if err != nil {
		t.Fatal(err)
	}
	z, err := gzip.NewReader(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	defer z.Close()
	data, err = io.ReadAll(io.LimitReader(z, 16<<20))
	if err != nil || len(data) >= 16<<20 {
		t.Fatal("invalid ResNet fixture", err)
	}
	var f struct {
		Schema    int
		Abs       float64 `json:"absolute_tolerance"`
		Rel       float64 `json:"relative_tolerance"`
		Cases     []resnetCase
		Reference struct {
			SHA string `json:"resnet_sha256"`
		}
	}
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	if f.Schema != 1 || f.Abs != 2e-6 || f.Rel != 2e-6 || len(f.Cases) != 3 || f.Reference.SHA != "2de7673e14e8c74d6c430e0b6e6cc844157f47ac35b67ff6238ca2d2c559eba9" {
		t.Fatal("ResNet fixture changed")
	}
	return f.Cases
}
func checkEmbedding(t *testing.T, result *WeSpeakerEmbeddingResult, want resnetMaskCase) {
	t.Helper()
	closePool(t, result.Embeddings, want.Embeddings)
	closePool(t, result.WeightSum, want.WeightSum)
	if !reflect.DeepEqual(result.NonzeroFrames, want.NonzeroFrames) {
		t.Fatal("mask support mismatch", result.NonzeroFrames, want.NonzeroFrames)
	}
}

func TestWeSpeakerResNetFullDepthOracle(t *testing.T) {
	for _, c := range loadResNetFixtures(t) {
		m, err := NewWeSpeakerResNet34(context.Background(), c.Config, c.Weights)
		if err != nil {
			t.Fatal(err)
		}
		for _, mode := range []WeSpeakerBlockMode{WeSpeakerBlockScalar, WeSpeakerBlockSIMD} {
			seen := 0
			features, shape, err := m.ForwardFramesObserved(context.Background(), c.Input, c.Frames, mode, func(stage, block int, shape CHWShape, x []float32) {
				b := c.Boundaries[seen]
				seen++
				if stage != b.Stage || block != b.Block || shape != b.Shape {
					t.Fatal("trunk boundary layout/order")
				}
				closePool(t, x, b.Values)
			})
			if err != nil {
				t.Fatal(err)
			}
			if seen != 17 {
				t.Fatal("not full ResNet34 depth", seen)
			}
			if shape != (CHWShape{8 * c.Config.BaseChannels, c.Config.MelBins / 8, (c.Frames + 7) / 8}) {
				t.Fatal("terminal layout", shape)
			}
			for _, mc := range c.MaskCases {
				seen := 0
				result, err := m.ForwardEmbeddingObserved(context.Background(), features, shape, mc.Masks, mc.Speakers, mc.MaskFrames, mode, func(stage string, rows, width int, x []float32) {
					if rows != max(1, mc.Speakers) || len(x) != rows*width {
						t.Fatal("pool/projection layout")
					}
					if seen == 0 && stage == "stats" {
						closePool(t, x, mc.Statistics)
					} else if seen == 1 && stage == "embedding" {
						closePool(t, x, mc.Embeddings)
					} else {
						t.Fatal("embedding boundary order")
					}
					seen++
				})
				if err != nil {
					t.Fatal(err)
				}
				if seen != 2 {
					t.Fatal("missing embedding boundary")
				}
				checkEmbedding(t, result, mc)
				combined, err := m.Forward(context.Background(), c.Input, c.Frames, mc.Masks, mc.Speakers, mc.MaskFrames, mode)
				if err != nil {
					t.Fatal(err)
				}
				checkEmbedding(t, combined, mc)
				if !reflect.DeepEqual(combined, result) {
					t.Fatal("shared trunk and combined path differ")
				}
			}
		}
	}
}

func TestWeSpeakerResNetTrustedChainMatchesObservedAndRetainsObserverGuard(t *testing.T) {
	c := loadResNetFixtures(t)[1]
	m, err := NewWeSpeakerResNet34(context.Background(), c.Config, c.Weights)
	if err != nil {
		t.Fatal(err)
	}
	for _, mode := range []WeSpeakerBlockMode{WeSpeakerBlockScalar, WeSpeakerBlockSIMD} {
		trusted, trustedShape, err := m.ForwardFrames(context.Background(), c.Input, c.Frames, mode)
		if err != nil {
			t.Fatal(err)
		}
		observed, observedShape, err := m.ForwardFramesObserved(context.Background(), c.Input, c.Frames, mode, func(int, int, CHWShape, []float32) {})
		if err != nil || trustedShape != observedShape || !reflect.DeepEqual(trusted, observed) {
			t.Fatal("trusted chain changed output", mode, err)
		}
	}
	seen := 0
	out, shape, err := m.ForwardFramesObserved(context.Background(), c.Input, c.Frames, WeSpeakerBlockSIMD, func(_, _ int, _ CHWShape, values []float32) {
		if seen == 0 {
			values[0] = float32(math.NaN()) // forbidden callback mutation
		}
		seen++
	})
	if err == nil || out != nil || shape != (CHWShape{}) || seen != 1 {
		t.Fatal("nonfinite observer mutation reached next boundary", seen, err)
	}
}

func TestWeSpeakerResNetMaskReuseOwnershipAndConcurrency(t *testing.T) {
	c := loadResNetFixtures(t)[0]
	m, err := NewWeSpeakerResNet34(context.Background(), c.Config, c.Weights)
	if err != nil {
		t.Fatal(err)
	}
	for _, x := range [][]float32{c.Weights.Stem, c.Weights.StemBN.RunningMean, c.Weights.Stages[2][3].Conv1, c.Weights.Projection.Weight, c.Weights.Projection.Bias} {
		for i := range x {
			x[i] = 99
		}
	}
	before := append([]float32(nil), c.Input...)
	features, shape, err := m.ForwardFrames(context.Background(), c.Input, c.Frames, WeSpeakerBlockSIMD)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(before, c.Input) {
		t.Fatal("Fbank mutated")
	}
	mc := c.MaskCases[1]
	original := append([]float32(nil), features...)
	maskCopy := append([]float32(nil), mc.Masks...)
	multi, err := m.ForwardEmbedding(context.Background(), features, shape, mc.Masks, mc.Speakers, mc.MaskFrames, WeSpeakerBlockSIMD)
	if err != nil {
		t.Fatal(err)
	}
	checkEmbedding(t, multi, mc)
	// Invalidate the trunk in this package-only test: reuse must not call it.
	stem := m.stem
	m.stem = nil
	for mask := 0; mask < mc.Speakers; mask++ {
		one, err := m.ForwardEmbedding(context.Background(), features, shape, mc.Masks[mask*mc.MaskFrames:(mask+1)*mc.MaskFrames], 1, mc.MaskFrames, WeSpeakerBlockSIMD)
		if err != nil {
			t.Fatal(err)
		}
		closePool(t, one.Embeddings, multi.Embeddings[mask*c.Config.EmbedDim:(mask+1)*c.Config.EmbedDim])
	}
	m.stem = stem
	if !reflect.DeepEqual(features, original) || !reflect.DeepEqual(maskCopy, mc.Masks) {
		t.Fatal("shared inputs mutated")
	}
	// Empty mask projects zero stats to the BIAS, not a valid embedding flag.
	if multi.NonzeroFrames[1] != 0 {
		t.Fatal("empty support hidden")
	}
	closePool(t, multi.Embeddings[c.Config.EmbedDim:2*c.Config.EmbedDim], m.projection.Bias)
	multi.Embeddings[0] = 99
	multi.WeightSum[0] = 99
	multi.NonzeroFrames[0] = 99
	var wg sync.WaitGroup
	for i := 0; i < 3; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			out, err := m.Forward(context.Background(), c.Input, c.Frames, mc.Masks, mc.Speakers, mc.MaskFrames, WeSpeakerBlockSIMD)
			if err != nil {
				t.Error(err)
				return
			}
			checkEmbedding(t, out, mc)
		}()
	}
	wg.Wait()
}

func TestWeSpeakerResNetRejectsGeometryAndWeights(t *testing.T) {
	for _, cfg := range []WeSpeakerResNetConfig{{}, {33, 80, 256}, {1, 7, 2}, {1, 81, 2}, {1, 10, 2}, {1, 8, 0}, {1, 8, 513}, {int(^uint(0) >> 1), 8, 2}} {
		if m, err := NewWeSpeakerResNet34(context.Background(), cfg, WeSpeakerResNetWeights{}); m != nil || err == nil {
			t.Fatal("bad config")
		}
	}
	fixture := loadResNetFixtures(t)[0]
	for _, kind := range []string{"stage_count", "stem", "bn", "variance", "projection", "late_block", "nan"} {
		bytes, _ := json.Marshal(fixture.Weights)
		var w WeSpeakerResNetWeights
		_ = json.Unmarshal(bytes, &w)
		switch kind {
		case "stage_count":
			w.Stages[2] = w.Stages[2][:5]
		case "stem":
			w.Stem = nil
		case "bn":
			w.StemBN.Bias = nil
		case "variance":
			w.StemBN.RunningVariance[0] = -1
		case "projection":
			w.Projection.Weight = nil
		case "late_block":
			w.Stages[3][2].Conv1 = nil
		case "nan":
			w.Projection.Bias[0] = float32(math.NaN())
		}
		if m, err := NewWeSpeakerResNet34(context.Background(), fixture.Config, w); m != nil || err == nil {
			t.Fatal("bad weights", kind)
		}
	}
	c := fixture
	m, err := NewWeSpeakerResNet34(context.Background(), c.Config, c.Weights)
	if err != nil {
		t.Fatal(err)
	}
	for _, frames := range []int{-1, 0, 4097, int(^uint(0) >> 1)} {
		if _, err := m.FrameShape(frames); err == nil {
			t.Fatal("invalid frames")
		}
	}
	for _, frames := range []int{1, 8, 9, 16, 17, 4096} {
		shape, err := m.FrameShape(frames)
		if err != nil {
			t.Fatal(err)
		}
		if shape.Frames != (frames+7)/8 {
			t.Fatal("wrong temporal stride")
		}
	}
	if out, _, err := m.ForwardFrames(context.Background(), c.Input[:len(c.Input)-1], c.Frames, WeSpeakerBlockScalar); err == nil || out != nil {
		t.Fatal("short Fbank")
	}
	if _, _, err := m.ForwardFrames(context.Background(), c.Input, c.Frames, WeSpeakerBlockMode(7)); err == nil {
		t.Fatal("invalid mode")
	}
	input := append([]float32(nil), c.Input...)
	input[0] = float32(math.Inf(1))
	if _, _, err := m.ForwardFrames(context.Background(), input, c.Frames, WeSpeakerBlockSIMD); err == nil {
		t.Fatal("bad Fbank")
	}
	features, shape, err := m.ForwardFrames(context.Background(), c.Input, c.Frames, WeSpeakerBlockSIMD)
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		masks                []float32
		speakers, maskFrames int
	}{{nil, 1, 1}, {[]float32{}, 1, 1}, {[]float32{-1}, 1, 1}, {[]float32{1}, 9, 1}, {[]float32{1}, 1, 4097}, {[]float32{float32(math.NaN())}, 1, 1}} {
		if out, err := m.ForwardEmbedding(context.Background(), features, shape, test.masks, test.speakers, test.maskFrames, WeSpeakerBlockSIMD); out != nil || err == nil {
			t.Fatal("invalid mask")
		}
		if out, err := m.Forward(context.Background(), nil, c.Frames, test.masks, test.speakers, test.maskFrames, WeSpeakerBlockSIMD); out != nil || err == nil {
			t.Fatal("invalid combined mask")
		}
	}
	// Public trunk-reuse buffers are checked by StatsPool before statistics or
	// observers. Verify the composed contract, not only StatsPool in isolation.
	for _, kind := range []string{"short", "oversized", "nan", "inf"} {
		bad := append([]float32(nil), features...)
		switch kind {
		case "short":
			bad = bad[:len(bad)-1]
		case "oversized":
			bad = append(bad, 0)
		case "nan":
			bad[0] = float32(math.NaN())
		case "inf":
			bad[len(bad)-1] = float32(math.Inf(1))
		}
		out, err := m.ForwardEmbeddingObserved(context.Background(), bad, shape, nil, 0, 0, WeSpeakerBlockSIMD, func(string, int, int, []float32) { t.Fatal("invalid reused feature buffer reached observer") })
		if err == nil || out != nil {
			t.Fatal("invalid reused feature buffer accepted", kind, err)
		}
	}
	// A zero-length unweighted temporal variance is not silently invented.
	if _, err := m.ForwardEmbedding(context.Background(), features[:shape.Channels*shape.Frequency], CHWShape{shape.Channels, shape.Frequency, 1}, nil, 0, 0, WeSpeakerBlockSIMD); err == nil {
		t.Fatal("single unweighted frame")
	}
	if _, err := m.ForwardEmbedding(context.Background(), features, CHWShape{1, 1, shape.Frames}, nil, 0, 0, WeSpeakerBlockSIMD); err == nil {
		t.Fatal("invalid CHW binding")
	}
	var missing *WeSpeakerResNet34
	var zero WeSpeakerResNet34
	for _, model := range []*WeSpeakerResNet34{missing, &zero} {
		if _, err := model.FrameShape(1); err == nil {
			t.Fatal("zero model")
		}
		if _, err := model.ForwardEmbedding(context.Background(), nil, shape, nil, 0, 0, WeSpeakerBlockSIMD); err == nil {
			t.Fatal("zero embedding model")
		}
	}
}

func TestWeSpeakerResNetCancellationAndSharedAllocation(t *testing.T) {
	c := loadResNetFixtures(t)[0]
	ctx := newPowersetContext(0)
	m, err := NewWeSpeakerResNet34(ctx, c.Config, c.Weights)
	ctx.cancel()
	if err != nil {
		t.Fatal(err)
	}
	// Sample constructor checkpoints throughout all16 blocks without timing sleeps.
	for at := 1; at <= ctx.calls; at += max(1, ctx.calls/25) {
		p := newPowersetContext(at)
		out, err := NewWeSpeakerResNet34(p, c.Config, c.Weights)
		p.cancel()
		if out != nil || !errors.Is(err, context.Canceled) {
			t.Fatal("construction cancellation", at)
		}
	}
	for stop := 0; stop < 17; stop++ {
		p, cancel := context.WithCancel(context.Background())
		seen := 0
		out, shape, err := m.ForwardFramesObserved(p, c.Input, c.Frames, WeSpeakerBlockSIMD, func(_, _ int, _ CHWShape, _ []float32) {
			if seen == stop {
				cancel()
			}
			seen++
		})
		cancel()
		if out != nil || shape != (CHWShape{}) || !errors.Is(err, context.Canceled) || seen != stop+1 {
			t.Fatal("trunk callback cancellation")
		}
	}
	features, shape, err := m.ForwardFrames(context.Background(), c.Input, c.Frames, WeSpeakerBlockSIMD)
	if err != nil {
		t.Fatal(err)
	}
	mc := c.MaskCases[1]
	count := newPowersetContext(0)
	_, err = m.ForwardEmbedding(count, features, shape, mc.Masks, mc.Speakers, mc.MaskFrames, WeSpeakerBlockSIMD)
	count.cancel()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("embedding checkpoints: %d", count.calls)
	for at := 1; at <= count.calls; at++ {
		p := newPowersetContext(at)
		out, err := m.ForwardEmbedding(p, features, shape, mc.Masks, mc.Speakers, mc.MaskFrames, WeSpeakerBlockSIMD)
		p.cancel()
		if out != nil || !errors.Is(err, context.Canceled) {
			t.Fatal("embedding cancellation", at, err)
		}
	}
	for _, target := range []string{"stats", "embedding"} {
		p, cancel := context.WithCancel(context.Background())
		out, err := m.ForwardEmbeddingObserved(p, features, shape, mc.Masks, mc.Speakers, mc.MaskFrames, WeSpeakerBlockSIMD, func(stage string, _, _ int, _ []float32) {
			if stage == target {
				cancel()
			}
		})
		cancel()
		if out != nil || !errors.Is(err, context.Canceled) {
			t.Fatal("embedding observer cancellation")
		}
	}
	var retained *WeSpeakerEmbeddingResult
	var lastErr error
	one := testing.AllocsPerRun(5, func() {
		retained, lastErr = m.ForwardEmbedding(context.Background(), features, shape, mc.Masks[:mc.MaskFrames], 1, mc.MaskFrames, WeSpeakerBlockSIMD)
	})
	many := testing.AllocsPerRun(5, func() {
		retained, lastErr = m.ForwardEmbedding(context.Background(), features, shape, mc.Masks, mc.Speakers, mc.MaskFrames, WeSpeakerBlockSIMD)
	})
	if retained == nil || lastErr != nil || one != many {
		t.Fatal("allocation count grows per mask", one, many, lastErr)
	}
	t.Logf("embedding allocations per call: %.0f", many)
	// Combined path's invalid masks are rejected before touching NaN Fbank.
	invalid := append([]float32(nil), c.Input...)
	invalid[0] = float32(math.NaN())
	p, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := m.Forward(p, invalid, c.Frames, nil, 0, 0, WeSpeakerBlockSIMD); !errors.Is(err, context.Canceled) {
		t.Fatal("combined precancel")
	}
}
