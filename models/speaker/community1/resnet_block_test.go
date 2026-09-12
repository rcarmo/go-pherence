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

type blockCase struct {
	Config        WeSpeakerBlockConfig
	Shape         CHWShape
	Weights       WeSpeakerBlockWeights
	Input, Output []float32
	Boundaries    []struct {
		Stage  string
		Shape  CHWShape
		Values []float32
	}
}

func loadBlockFixtures(t *testing.T) []blockCase {
	t.Helper()
	data, err := os.ReadFile("testdata/resnet-block-reference.json.gz")
	if err != nil {
		t.Fatal(err)
	}
	reader, err := gzip.NewReader(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()
	data, err = io.ReadAll(io.LimitReader(reader, 8<<20))
	if err != nil || len(data) >= 8<<20 {
		t.Fatal("invalid fixture", err)
	}
	var f struct {
		Schema    int
		Abs       float64 `json:"absolute_tolerance"`
		Rel       float64 `json:"relative_tolerance"`
		Reference struct {
			SHA string `json:"sha256"`
		}
		Cases []blockCase
	}
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	if f.Schema != 1 || f.Abs != 2e-6 || f.Rel != 2e-6 || len(f.Cases) != 6 || f.Reference.SHA != "2de7673e14e8c74d6c430e0b6e6cc844157f47ac35b67ff6238ca2d2c559eba9" {
		t.Fatal("BasicBlock fixture changed")
	}
	return f.Cases
}

func TestWeSpeakerBlockPinnedTorchOracle(t *testing.T) {
	for _, c := range loadBlockFixtures(t) {
		block, err := NewWeSpeakerBasicBlock(context.Background(), c.Config, c.Weights)
		if err != nil {
			t.Fatal(err)
		}
		for _, mode := range []WeSpeakerBlockMode{WeSpeakerBlockScalar, WeSpeakerBlockSIMD} {
			visited := 0
			result, shape, err := block.ForwardObserved(context.Background(), c.Input, c.Shape, mode, func(stage string, shape CHWShape, values []float32) {
				if visited >= len(c.Boundaries) {
					t.Fatal("extra boundary")
				}
				want := c.Boundaries[visited]
				visited++
				if stage != want.Stage || shape != want.Shape {
					t.Fatalf("boundary geometry/order %s %+v want%s %+v", stage, shape, want.Stage, want.Shape)
				}
				closePool(t, values, want.Values)
			})
			if err != nil {
				t.Fatal(err)
			}
			expected := CHWShape{c.Config.OutChannels, (c.Shape.Frequency + c.Config.Stride - 1) / c.Config.Stride, (c.Shape.Frames + c.Config.Stride - 1) / c.Config.Stride}
			if shape != expected || visited != len(c.Boundaries) {
				t.Fatal("wrong output geometry/boundaries")
			}
			closePool(t, result, c.Output)
		}
	}
}

func identityBN(channels int) WeSpeakerBN {
	b := WeSpeakerBN{Weight: make([]float32, channels), Bias: make([]float32, channels), RunningMean: make([]float32, channels), RunningVariance: make([]float32, channels)}
	for i := range b.Weight {
		b.Weight[i] = 1
		b.RunningVariance[i] = 1
	}
	return b
}
func centerIdentityBlock() WeSpeakerBlockWeights {
	w := WeSpeakerBlockWeights{Conv1: make([]float32, 9), Conv2: make([]float32, 9), BN1: identityBN(1), BN2: identityBN(1)}
	w.Conv1[4] = 1
	w.Conv2[4] = 1
	return w
}

func TestWeSpeakerBlockPaddingIdentityAndBatchNorm(t *testing.T) {
	w := centerIdentityBlock()
	shape := CHWShape{1, 2, 3}
	input := []float32{-3, 2, 1, 4, -5, 6}
	block, err := NewWeSpeakerBasicBlock(context.Background(), WeSpeakerBlockConfig{1, 1, 1}, w)
	if err != nil {
		t.Fatal(err)
	}
	result, _, err := block.Forward(context.Background(), input, shape, WeSpeakerBlockScalar)
	if err != nil {
		t.Fatal(err)
	}
	for i, value := range input {
		want := math.Max(float64(value), 0)/(1+1e-5) + float64(value)
		want = math.Max(want, 0)
		if math.Abs(float64(result[i])-want) > 2e-6 {
			t.Fatalf("identity residual[%d]: %g want%g", i, result[i], want)
		}
	}
	// Top-left kernel selects (f-1,t-1): proves both padding and temporal axis.
	w = centerIdentityBlock()
	w.Conv1[4] = 0
	w.Conv1[0] = 1
	block, err = NewWeSpeakerBasicBlock(context.Background(), WeSpeakerBlockConfig{1, 1, 1}, w)
	if err != nil {
		t.Fatal(err)
	}
	positive := []float32{1, 2, 3, 4, 5, 6}
	_, _, err = block.ForwardObserved(context.Background(), positive, shape, WeSpeakerBlockSIMD, func(stage string, _ CHWShape, x []float32) {
		if stage == "conv1" {
			closePool(t, x, []float32{0, 0, 0, 0, 1, 2})
		}
	})
	if err != nil {
		t.Fatal(err)
	}
	// Running statistics (not current input statistics) and zero variance + eps.
	bn := WeSpeakerBN{Weight: []float32{2, -1}, Bias: []float32{.5, .25}, RunningMean: []float32{3, -2}, RunningVariance: []float32{4, 0}}
	x := []float32{3, 5, -2, -1}
	want := []float32{.5, float32(4/math.Sqrt(4+1e-5) + .5), .25, float32(-1/math.Sqrt(1e-5) + .25)}
	if err := weSpeakerBlockBN(context.Background(), x, CHWShape{2, 1, 2}, bn); err != nil {
		t.Fatal(err)
	}
	closePool(t, x, want)
}

func TestWeSpeakerBlockRejectsMalformed(t *testing.T) {
	for _, cfg := range []WeSpeakerBlockConfig{{}, {1, 1, 0}, {1, 1, 3}, {257, 1, 1}, {1, 257, 1}, {int(^uint(0) >> 1), 1, 1}} {
		if block, err := NewWeSpeakerBasicBlock(context.Background(), cfg, WeSpeakerBlockWeights{}); err == nil || block != nil {
			t.Fatal("invalid config")
		}
	}
	fixtures := loadBlockFixtures(t)
	for _, kind := range []string{"conv1", "conv2", "BNweight", "BNbias", "mean", "variance", "negativeVariance", "nan", "inf", "shortcut", "shortcutBN"} {
		encoded, _ := json.Marshal(fixtures[1].Weights)
		var w WeSpeakerBlockWeights
		_ = json.Unmarshal(encoded, &w)
		switch kind {
		case "conv1":
			w.Conv1 = nil
		case "conv2":
			w.Conv2 = nil
		case "BNweight":
			w.BN1.Weight = nil
		case "BNbias":
			w.BN2.Bias = nil
		case "mean":
			w.BN1.RunningMean = nil
		case "variance":
			w.BN2.RunningVariance = nil
		case "negativeVariance":
			w.BN1.RunningVariance[0] = -1
		case "nan":
			w.Conv1[0] = float32(math.NaN())
		case "inf":
			w.BN2.Bias[0] = float32(math.Inf(1))
		case "shortcut":
			w.Shortcut = nil
		case "shortcutBN":
			w.ShortcutBN.Bias = nil
		}
		if block, err := NewWeSpeakerBasicBlock(context.Background(), fixtures[1].Config, w); err == nil || block != nil {
			t.Fatal("bad weights", kind)
		}
	}
	w := centerIdentityBlock()
	w.Shortcut = []float32{1}
	if _, err := NewWeSpeakerBasicBlock(context.Background(), WeSpeakerBlockConfig{1, 1, 1}, w); err == nil {
		t.Fatal("unexpected shortcut")
	}
	c := fixtures[0]
	b, err := NewWeSpeakerBasicBlock(context.Background(), c.Config, c.Weights)
	if err != nil {
		t.Fatal(err)
	}
	for _, shape := range []CHWShape{{}, {2, 0, 1}, {2, 1, 0}, {2, 81, 1}, {2, 1, 4097}, {1, 1, 1}, {2, int(^uint(0) >> 1), 1}} {
		if _, err := b.OutputShape(shape); err == nil {
			t.Fatal("bad shape")
		}
	}
	if _, err := chwElements(CHWShape{256, 80, 4096}); err == nil {
		t.Fatal("element bound")
	}
	if out, shape, err := b.Forward(context.Background(), c.Input[:len(c.Input)-1], c.Shape, WeSpeakerBlockScalar); err == nil || out != nil || shape != (CHWShape{}) {
		t.Fatal("short input")
	}
	if _, _, err := b.Forward(context.Background(), c.Input, c.Shape, WeSpeakerBlockMode(9)); err == nil {
		t.Fatal("invalid mode")
	}
	c.Input[0] = float32(math.NaN())
	if _, _, err := b.Forward(context.Background(), c.Input, c.Shape, WeSpeakerBlockScalar); err == nil {
		t.Fatal("NaN input")
	}
	var missing *WeSpeakerBasicBlock
	var zero WeSpeakerBasicBlock
	for _, b := range []*WeSpeakerBasicBlock{missing, &zero} {
		if out, shape, err := b.Forward(context.Background(), nil, CHWShape{1, 1, 1}, WeSpeakerBlockSIMD); err == nil || out != nil || shape != (CHWShape{}) {
			t.Fatal("zero model")
		}
	}
	w = centerIdentityBlock()
	w.Conv1[4] = math.MaxFloat32
	b, err = NewWeSpeakerBasicBlock(context.Background(), WeSpeakerBlockConfig{1, 1, 1}, w)
	if err != nil {
		t.Fatal(err)
	}
	for _, mode := range []WeSpeakerBlockMode{WeSpeakerBlockScalar, WeSpeakerBlockSIMD} {
		if out, _, err := b.Forward(context.Background(), []float32{math.MaxFloat32}, CHWShape{1, 1, 1}, mode); out != nil || err == nil {
			t.Fatal("overflow escaped")
		}
	}
}

func TestWeSpeakerBlockOwnershipAndConcurrency(t *testing.T) {
	c := loadBlockFixtures(t)[1]
	block, err := NewWeSpeakerBasicBlock(context.Background(), c.Config, c.Weights)
	if err != nil {
		t.Fatal(err)
	}
	for _, values := range [][]float32{c.Weights.Conv1, c.Weights.Conv2, c.Weights.Shortcut, c.Weights.BN1.RunningMean, c.Weights.BN2.RunningVariance, c.Weights.ShortcutBN.Weight} {
		for i := range values {
			values[i] = 99
		}
	}
	before := append([]float32(nil), c.Input...)
	out, _, err := block.Forward(context.Background(), c.Input, c.Shape, WeSpeakerBlockSIMD)
	if err != nil {
		t.Fatal(err)
	}
	closePool(t, out, c.Output)
	out[0] = 999
	if !reflect.DeepEqual(before, c.Input) {
		t.Fatal("mutated input")
	}
	var wg sync.WaitGroup
	for i := 0; i < 3; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			out, _, err := block.Forward(context.Background(), c.Input, c.Shape, WeSpeakerBlockSIMD)
			if err != nil {
				t.Error(err)
				return
			}
			closePool(t, out, c.Output)
		}()
	}
	wg.Wait()
	// Identity shortcut never aliases the returned tensor to input either.
	c = loadBlockFixtures(t)[0]
	block, err = NewWeSpeakerBasicBlock(context.Background(), c.Config, c.Weights)
	if err != nil {
		t.Fatal(err)
	}
	out, _, err = block.Forward(context.Background(), c.Input, c.Shape, WeSpeakerBlockSIMD)
	if err != nil {
		t.Fatal(err)
	}
	first := c.Input[0]
	out[0] = 99
	if c.Input[0] != first {
		t.Fatal("identity shortcut output aliases input")
	}
}

func TestWeSpeakerBlockCancellationAndAllocationBound(t *testing.T) {
	c := loadBlockFixtures(t)[4] // small identity case for exhaustive cancellation
	ctx := newPowersetContext(0)
	block, err := NewWeSpeakerBasicBlock(ctx, c.Config, c.Weights)
	ctx.cancel()
	if err != nil {
		t.Fatal(err)
	}
	for at := 1; at <= ctx.calls; at++ {
		ctx := newPowersetContext(at)
		out, err := NewWeSpeakerBasicBlock(ctx, c.Config, c.Weights)
		ctx.cancel()
		if out != nil || !errors.Is(err, context.Canceled) {
			t.Fatal("constructor cancel", at, err)
		}
	}
	for _, mode := range []WeSpeakerBlockMode{WeSpeakerBlockScalar, WeSpeakerBlockSIMD} {
		count := newPowersetContext(0)
		_, _, err := block.Forward(count, c.Input, c.Shape, mode)
		count.cancel()
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("block cancel checkpoints mode%d: %d", mode, count.calls)
		for at := 1; at <= count.calls; at++ {
			ctx := newPowersetContext(at)
			out, shape, err := block.Forward(ctx, c.Input, c.Shape, mode)
			ctx.cancel()
			if out != nil || shape != (CHWShape{}) || !errors.Is(err, context.Canceled) {
				t.Fatal("forward cancellation", at, err)
			}
		}
	}
	c = loadBlockFixtures(t)[1]
	block, err = NewWeSpeakerBasicBlock(context.Background(), c.Config, c.Weights)
	if err != nil {
		t.Fatal(err)
	}
	for target := range c.Boundaries {
		ctx, cancel := context.WithCancel(context.Background())
		seen := 0
		out, shape, err := block.ForwardObserved(ctx, c.Input, c.Shape, WeSpeakerBlockSIMD, func(stage string, _ CHWShape, _ []float32) {
			if stage != c.Boundaries[seen].Stage {
				t.Fatal("bad observed order")
			}
			if seen == target {
				cancel()
			}
			seen++
		})
		cancel()
		if out != nil || shape != (CHWShape{}) || !errors.Is(err, context.Canceled) || seen != target+1 {
			t.Fatal("observer cancellation")
		}
	}
	var kept []float32
	var last error
	short := testing.AllocsPerRun(5, func() {
		kept, _, last = block.Forward(context.Background(), make([]float32, c.Config.InChannels), CHWShape{c.Config.InChannels, 1, 1}, WeSpeakerBlockSIMD)
	})
	tinyInput := make([]float32, c.Config.InChannels)
	tiny := testing.AllocsPerRun(5, func() {
		kept, _, last = block.Forward(context.Background(), tinyInput, CHWShape{c.Config.InChannels, 1, 1}, WeSpeakerBlockSIMD)
	})
	full := testing.AllocsPerRun(5, func() { kept, _, last = block.Forward(context.Background(), c.Input, c.Shape, WeSpeakerBlockSIMD) })
	if kept == nil || last != nil || tiny != full {
		t.Fatalf("per-position allocations tiny%g full%g %v", tiny, full, last)
	}
	t.Logf("block allocations per call: %.0f (caller input alloc excluded; control %.0f)", full, short)
}
