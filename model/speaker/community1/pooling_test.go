package community1

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"os"
	"reflect"
	"testing"
)

type poolCase struct {
	Config               StatsPoolConfig
	Input, Masks, Output []float32
	WeightSum            []float32 `json:"weight_sum"`
	NonzeroFrames        []int     `json:"nonzero_frames"`
}

func loadPoolFixtures(t *testing.T) []poolCase {
	t.Helper()
	data, err := os.ReadFile("testdata/pooling-reference.json")
	if err != nil {
		t.Fatal(err)
	}
	var f struct {
		Schema    int
		Abs       float64 `json:"absolute_tolerance"`
		Rel       float64 `json:"relative_tolerance"`
		Cases     []poolCase
		Reference struct {
			SHA string `json:"sha256"`
		}
	}
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	if f.Schema != 1 || f.Abs != 2e-6 || f.Rel != 2e-6 || len(f.Cases) != 9 || f.Reference.SHA != "8cb687441630e6759fb6ca545d649b41dbec1d42f869954c0a03ae191c7cbd82" {
		t.Fatal("pool fixture changed")
	}
	return f.Cases
}
func closePool(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatal("pool output shape")
	}
	for i, value := range got {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || math.Abs(float64(value-want[i])) > 2e-6+2e-6*math.Abs(float64(want[i])) {
			t.Fatalf("pool[%d] got%.10g want%.10g", i, value, want[i])
		}
	}
}

func TestTorchSumF32PinnedReduction(t *testing.T) {
	for _, tc := range []struct {
		name string
		in   []float32
		want uint32
	}{
		{"five", []float32{1, .3, 0, .8, 1}, 0x40466666},
		{"thirteen", []float32{1, 1, .3, .3, 0, 0, .8, .8, 1, 1, .2, .2, .7}, 0x40e9999c},
		{"sixty-three", func() []float32 {
			base := []float32{1, .3, 0, .8, 1, .2, .7}
			out := make([]float32, 63)
			for i := range out {
				out[i] = base[poolMaskIndex(i, len(base), len(out))]
			}
			return out
		}(), 0x42100000},
		{"large-tail", func() []float32 {
			out := make([]float32, 4095)
			for i := range out {
				value := float32((int64(i)*1103515245 + 12345) % 65536)
				value = value / float32(32768)
				value = value - float32(1)
				out[i] = value * float32(.2)
			}
			return out
		}(), 0xbed4a32e},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := math.Float32bits(torchSumF32(tc.in)); got != tc.want {
				t.Fatalf("torch sum bits=%08x want=%08x", got, tc.want)
			}
		})
	}
}

func TestStatsPoolPinnedOracle(t *testing.T) {
	for _, c := range loadPoolFixtures(t) {
		input := append([]float32(nil), c.Input...)
		var masks []float32
		if c.Masks != nil {
			masks = append([]float32{}, c.Masks...)
		}
		result, err := StatsPool(context.Background(), c.Input, c.Masks, c.Config)
		if err != nil {
			t.Fatal(err)
		}
		closePool(t, result.Statistics, c.Output)
		closePool(t, result.WeightSum, c.WeightSum)
		for i, value := range result.WeightSum {
			if math.Float32bits(value) != math.Float32bits(c.WeightSum[i]) {
				t.Fatalf("weight sum[%d] bits=%08x want=%08x", i, math.Float32bits(value), math.Float32bits(c.WeightSum[i]))
			}
		}
		if !reflect.DeepEqual(result.NonzeroFrames, c.NonzeroFrames) {
			t.Fatal("support", result.NonzeroFrames, c.NonzeroFrames)
		}
		if !reflect.DeepEqual(input, c.Input) || !reflect.DeepEqual(masks, c.Masks) {
			t.Fatal("input/mask mutated")
		}
		if c.Config.Speakers > 1 {
			for speaker := 0; speaker < c.Config.Speakers; speaker++ {
				cfg := c.Config
				cfg.Speakers = 1
				single, err := StatsPool(context.Background(), c.Input, c.Masks[speaker*cfg.MaskFrames:(speaker+1)*cfg.MaskFrames], cfg)
				if err != nil {
					t.Fatal(err)
				}
				closePool(t, single.Statistics, result.Statistics[speaker*2*cfg.Features:(speaker+1)*2*cfg.Features])
			}
		}
	}
}

func TestStatsPoolAnalyticMaskAndVarianceSemantics(t *testing.T) {
	cfg := StatsPoolConfig{Features: 2, Frames: 3}
	result, err := StatsPool(context.Background(), []float32{1, 2, 3, 5, 5, 5}, nil, cfg)
	if err != nil {
		t.Fatal(err)
	}
	closePool(t, result.Statistics, []float32{2, 5, 1, 0}) // unbiased std, all means then stds
	// Selecting frames0 and2 yields sample std sqrt(2), not population std1.
	cfg.Speakers = 3
	cfg.MaskFrames = 3
	result, err = StatsPool(context.Background(), []float32{1, 2, 3, 5, 5, 5}, []float32{1, 0, 1, 0, 0, 0, 0, 1, 0}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	closePool(t, result.Statistics, []float32{2, 5, float32(math.Sqrt2), 0, 0, 0, 0, 0, 2, 5, 0, 0})
	if !reflect.DeepEqual(result.NonzeroFrames, []int{2, 0, 1}) {
		t.Fatal("empty/single support was hidden")
	}
	// Nearest legacy floor indices: masks[0,1] upsampled to5 => [0,0,0,1,1].
	cfg = StatsPoolConfig{Features: 1, Frames: 5, Speakers: 1, MaskFrames: 2}
	result, err = StatsPool(context.Background(), []float32{1, 2, 3, 4, 5}, []float32{0, 1}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	closePool(t, result.Statistics, []float32{4.5, float32(math.Sqrt(.5))})
	if result.NonzeroFrames[0] != 2 {
		t.Fatal("not nearest legacy resize")
	}
	// Downsample8->3 picks0,2,5 (neither center alignment nor nearest-exact).
	cfg = StatsPoolConfig{Features: 1, Frames: 3, Speakers: 1, MaskFrames: 8}
	result, err = StatsPool(context.Background(), []float32{1, 2, 3}, []float32{1, 0, 0, 0, 0, 1, 0, 0}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	closePool(t, result.Statistics, []float32{2, float32(math.Sqrt2)})
}

func TestStatsPoolBoundsInvalidMasksAndOverflow(t *testing.T) {
	for _, cfg := range []StatsPoolConfig{{}, {1, 0, 0, 0}, {4097, 1, 0, 0}, {1, 4097, 0, 0}, {int(^uint(0) >> 1), 2, 0, 0}, {1, 1, 0, 0}, {1, 2, 1, 0}, {1, 2, 0, 1}} {
		if result, err := StatsPool(context.Background(), nil, nil, cfg); err == nil || result != nil {
			t.Fatal("invalid unweighted geometry")
		}
	}
	for _, cfg := range []StatsPoolConfig{{1, 2, 0, 2}, {1, 2, 9, 2}, {1, 2, 1, 4097}, {1, 2, 1, 0}, {1, 2, -1, 2}, {1, 2, 1, 2}} {
		if result, err := StatsPool(context.Background(), []float32{1, 2}, []float32{}, cfg); err == nil || result != nil {
			t.Fatal("invalid masked geometry")
		}
	}
	for _, value := range []float32{-1, 1.1, float32(math.NaN()), float32(math.Inf(1))} {
		if out, err := StatsPool(context.Background(), []float32{1, 2}, []float32{1, value}, StatsPoolConfig{1, 2, 1, 2}); err == nil || out != nil {
			t.Fatal("invalid mask")
		}
	}
	for _, value := range []float32{float32(math.NaN()), float32(math.Inf(-1))} {
		if out, err := StatsPool(context.Background(), []float32{value, 1}, nil, StatsPoolConfig{1, 2, 0, 0}); err == nil || out != nil {
			t.Fatal("invalid feature")
		}
	}
	// Finite but unrepresentable std or float32 masked reductions are errors.
	for _, masks := range [][]float32{nil, {1, 1}} {
		cfg := StatsPoolConfig{1, 2, 0, 0}
		if masks != nil {
			cfg.Speakers = 1
			cfg.MaskFrames = 2
		}
		if out, err := StatsPool(context.Background(), []float32{math.MaxFloat32, -math.MaxFloat32}, masks, cfg); err == nil || out != nil {
			t.Fatal("nonfinite reduction escaped")
		}
	}
	// Geometry upper bounds without allocating maxFeatures*maxFrames: each axis
	// exercised independently; not a model workload or performance measurement.
	for _, cfg := range []StatsPoolConfig{{4096, 2, 0, 0}, {1, 4096, 0, 0}, {1, 2, 8, 4096}} {
		input := make([]float32, cfg.Features*cfg.Frames)
		var masks []float32
		if cfg.Speakers > 0 {
			masks = make([]float32, cfg.Speakers*cfg.MaskFrames)
		}
		if _, err := StatsPool(context.Background(), input, masks, cfg); err != nil {
			t.Fatal(err)
		}
	}
}

func TestStatsPoolCancellationAndOwnedResults(t *testing.T) {
	c := loadPoolFixtures(t)[2]
	count := newPowersetContext(0)
	out, err := StatsPool(count, c.Input, c.Masks, c.Config)
	count.cancel()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("masked pool cancellation checkpoints: %d", count.calls)
	for at := 1; at <= count.calls; at++ {
		ctx := newPowersetContext(at)
		result, err := StatsPool(ctx, c.Input, c.Masks, c.Config)
		ctx.cancel()
		if result != nil || !errors.Is(err, context.Canceled) {
			t.Fatal("partial output/cancellation", at, err)
		}
	}
	out.Statistics[0] = 99
	out.WeightSum[0] = 99
	out.NonzeroFrames[0] = 99
	again, err := StatsPool(context.Background(), c.Input, c.Masks, c.Config)
	if err != nil {
		t.Fatal(err)
	}
	closePool(t, again.Statistics, c.Output)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if result, err := StatsPool(ctx, nil, nil, StatsPoolConfig{}); result != nil || !errors.Is(err, context.Canceled) {
		t.Fatal("precancel")
	}
	// Shared features and only one reusable resized mask row: allocations do not
	// grow with number of masks; callback output count grows as intended.
	var kept *StatsPoolResult
	var lastErr error
	single := c.Config
	single.Speakers = 1
	one := testing.AllocsPerRun(5, func() {
		kept, lastErr = StatsPool(context.Background(), c.Input, c.Masks[:c.Config.MaskFrames], single)
	})
	multi := testing.AllocsPerRun(5, func() { kept, lastErr = StatsPool(context.Background(), c.Input, c.Masks, c.Config) })
	if lastErr != nil || kept == nil || one != multi {
		t.Fatalf("allocations grew by masks: %g %g %v", one, multi, lastErr)
	}
	t.Logf("pool allocations per call: %.0f", multi)
}

func TestStatsPoolTorchNearestIndexRounding(t *testing.T) {
	data, err := os.ReadFile("testdata/pooling-reference.json")
	if err != nil {
		t.Fatal(err)
	}
	var f struct {
		Cases []struct {
			Source, Target int
			Indexes        []int
		} `json:"index_cases"`
	}
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	if len(f.Cases) != 10 {
		t.Fatal("missing nearest index fixtures")
	}
	for _, c := range f.Cases {
		if len(c.Indexes) != c.Target {
			t.Fatal("index fixture shape")
		}
		for i, want := range c.Indexes {
			if got := poolMaskIndex(i, c.Source, c.Target); got != want {
				t.Fatalf("nearest%d->%d index%d: got%d want%d", c.Source, c.Target, i, got, want)
			}
		}
	}
	if got := poolMaskIndex(41, 2, 82); got != 0 {
		t.Fatal("used exact ratio instead of Torch f32 boundary")
	}
}
