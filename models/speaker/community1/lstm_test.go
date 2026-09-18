package community1

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"os"
	"reflect"
	"sync"
	"testing"
)

type lstmFixture struct {
	Config                      LSTMConfig
	Frames                      int
	Weights                     []LSTMLayer
	Input, Output, Hidden, Cell []float32
	InitialHidden               []float32   `json:"initial_hidden"`
	InitialCell                 []float32   `json:"initial_cell"`
	LayerOutputs                [][]float32 `json:"layer_outputs"`
}

func loadLSTMFixtures(t *testing.T) []lstmFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/lstm-reference.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Schema    int
		Tolerance float64 `json:"absolute_tolerance"`
		Reference struct {
			SHA string `json:"pyannet_sha256"`
		}
		Cases []lstmFixture
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Schema != 1 || fixture.Tolerance != 2e-6 || len(fixture.Cases) != 6 || fixture.Reference.SHA != "3ceebc8c00e83d96747a706212102e7e99c44732ff1fc769e241e8de47d3d6af" {
		t.Fatal("LSTM reference changed")
	}
	for i := range fixture.Cases {
		if len(fixture.Cases[i].InitialHidden) == 0 {
			fixture.Cases[i].InitialHidden = nil
			fixture.Cases[i].InitialCell = nil
		}
	}
	return fixture.Cases
}
func closeLSTM(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("LSTM shape got%d want%d", len(got), len(want))
	}
	for i, value := range got {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || math.Abs(float64(value-want[i])) > 2e-6 {
			t.Fatalf("LSTM element%d got%.9g want%.9g", i, value, want[i])
		}
	}
}

func TestLSTMPinnedTorchOracle(t *testing.T) {
	for index, fixture := range loadLSTMFixtures(t) {
		m, err := NewLSTM(context.Background(), fixture.Config, fixture.Weights)
		if err != nil {
			t.Fatal(err)
		}
		for _, mode := range []LSTMMode{LSTMScalar, LSTMSIMD} {
			observed := 0
			result, err := m.ForwardObserved(context.Background(), fixture.Input, fixture.Frames, fixture.InitialHidden, fixture.InitialCell, mode, func(layer, frames, width int, values []float32) {
				if layer != observed || frames != fixture.Frames || width != lstmDirections(fixture.Config)*fixture.Config.HiddenSize {
					t.Fatal("layer observer geometry/order")
				}
				closeLSTM(t, values, fixture.LayerOutputs[layer])
				observed++
			})
			if err != nil {
				t.Fatalf("case%d mode%d: %v", index, mode, err)
			}
			closeLSTM(t, result.Output, fixture.Output)
			closeLSTM(t, result.Hidden, fixture.Hidden)
			closeLSTM(t, result.Cell, fixture.Cell)
			if observed != fixture.Config.NumLayers {
				t.Fatal("missing layer observer")
			}
			// Top-layer terminal state is at last forward frame but FIRST reverse frame.
			h, dirs := fixture.Config.HiddenSize, lstmDirections(fixture.Config)
			top := (fixture.Config.NumLayers - 1) * dirs * h
			closeLSTM(t, result.Hidden[top:top+h], result.Output[(fixture.Frames-1)*dirs*h:(fixture.Frames-1)*dirs*h+h])
			if dirs == 2 {
				closeLSTM(t, result.Hidden[top+h:top+2*h], result.Output[h:2*h])
			}
		}
	}
}

func TestLSTMStateResetOwnershipAndConcurrentCalls(t *testing.T) {
	f := loadLSTMFixtures(t)[3]
	m, err := NewLSTM(context.Background(), f.Config, f.Weights)
	if err != nil {
		t.Fatal(err)
	}
	// Model does not borrow any caller weight/bias buffers in either direction.
	for i := range f.Weights {
		for _, w := range []LSTMWeights{f.Weights[i].Forward, f.Weights[i].Reverse} {
			for _, array := range [][]float32{w.WeightIH, w.WeightHH, w.BiasIH, w.BiasHH} {
				for j := range array {
					array[j] = 99
				}
			}
		}
	}
	beforeInput := append([]float32(nil), f.Input...)
	beforeH := append([]float32(nil), f.InitialHidden...)
	beforeC := append([]float32(nil), f.InitialCell...)
	a, err := m.Forward(context.Background(), f.Input, f.Frames, f.InitialHidden, f.InitialCell, LSTMSIMD)
	if err != nil {
		t.Fatal(err)
	}
	closeLSTM(t, a.Output, f.Output)
	a.Output[0] = 100
	a.Hidden[0] = 100
	a.Cell[0] = 100
	b, err := m.Forward(context.Background(), f.Input, f.Frames, f.InitialHidden, f.InitialCell, LSTMSIMD)
	if err != nil {
		t.Fatal(err)
	}
	closeLSTM(t, b.Output, f.Output)
	closeLSTM(t, b.Hidden, f.Hidden)
	if !reflect.DeepEqual(beforeInput, f.Input) || !reflect.DeepEqual(beforeH, f.InitialHidden) || !reflect.DeepEqual(beforeC, f.InitialCell) {
		t.Fatal("mutated input/state")
	}
	var wg sync.WaitGroup
	for i := 0; i < 3; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			result, err := m.Forward(context.Background(), f.Input, f.Frames, f.InitialHidden, f.InitialCell, LSTMSIMD)
			if err != nil {
				t.Error(err)
				return
			}
			closeLSTM(t, result.Output, f.Output)
		}()
	}
	wg.Wait()
	// For unidirectional inference only, split-sequence terminal state resumes.
	f = loadLSTMFixtures(t)[4]
	m, err = NewLSTM(context.Background(), f.Config, f.Weights)
	if err != nil {
		t.Fatal(err)
	}
	cut := 7
	prefix, err := m.Forward(context.Background(), f.Input[:cut*f.Config.InputSize], cut, f.InitialHidden, f.InitialCell, LSTMScalar)
	if err != nil {
		t.Fatal(err)
	}
	suffix, err := m.Forward(context.Background(), f.Input[cut*f.Config.InputSize:], f.Frames-cut, prefix.Hidden, prefix.Cell, LSTMScalar)
	if err != nil {
		t.Fatal(err)
	}
	combined := append(append([]float32(nil), prefix.Output...), suffix.Output...)
	closeLSTM(t, combined, f.Output)
	closeLSTM(t, suffix.Hidden, f.Hidden)
	closeLSTM(t, suffix.Cell, f.Cell)
}

func TestLSTMRejectsMalformedInputs(t *testing.T) {
	for _, cfg := range []LSTMConfig{{}, {513, 1, 1, false}, {1, 257, 1, false}, {1, 1, 5, false}, {1, 1, -1, false}, {int(^uint(0) >> 1), 1, 1, false}} {
		if m, err := NewLSTM(context.Background(), cfg, nil); err == nil || m != nil {
			t.Fatal("invalid geometry accepted")
		}
	}
	f := loadLSTMFixtures(t)[1]
	for _, kind := range []string{"layers", "input_weight", "hidden_weight", "input_bias", "hidden_bias", "reverse", "nan", "inf"} {
		clone := loadLSTMFixtures(t)[1]
		switch kind {
		case "layers":
			clone.Weights = nil
		case "input_weight":
			clone.Weights[0].Forward.WeightIH = nil
		case "hidden_weight":
			clone.Weights[0].Forward.WeightHH = nil
		case "input_bias":
			clone.Weights[0].Forward.BiasIH = nil
		case "hidden_bias":
			clone.Weights[0].Forward.BiasHH = nil
		case "reverse":
			clone.Weights[0].Reverse = LSTMWeights{}
		case "nan":
			clone.Weights[0].Forward.WeightIH[0] = float32(math.NaN())
		case "inf":
			clone.Weights[0].Reverse.BiasHH[0] = float32(math.Inf(-1))
		}
		if m, err := NewLSTM(context.Background(), clone.Config, clone.Weights); err == nil || m != nil {
			t.Fatalf("accepted %s", kind)
		}
	}
	clone := loadLSTMFixtures(t)[0]
	clone.Weights[0].Reverse = clone.Weights[0].Forward
	if _, err := NewLSTM(context.Background(), clone.Config, clone.Weights); err == nil {
		t.Fatal("unexpected reverse weights")
	}
	m, err := NewLSTM(context.Background(), f.Config, f.Weights)
	if err != nil {
		t.Fatal(err)
	}
	for _, frames := range []int{-1, 0, 4097, int(^uint(0) >> 1)} {
		if out, err := m.Forward(context.Background(), nil, frames, nil, nil, LSTMScalar); out != nil || err == nil {
			t.Fatal("bad frame count")
		}
	}
	for _, tc := range []struct {
		input, h, c []float32
		mode        LSTMMode
	}{
		{f.Input[:len(f.Input)-1], nil, nil, LSTMScalar}, {f.Input, []float32{0}, nil, LSTMScalar}, {f.Input, f.InitialHidden, nil, LSTMScalar}, {f.Input, nil, f.InitialCell, LSTMScalar}, {f.Input, nil, nil, LSTMMode(2)},
	} {
		if out, err := m.Forward(context.Background(), tc.input, f.Frames, tc.h, tc.c, tc.mode); err == nil || out != nil {
			t.Fatal("bad state/shape/mode")
		}
	}
	input := append([]float32(nil), f.Input...)
	input[0] = float32(math.NaN())
	if out, err := m.Forward(context.Background(), input, f.Frames, nil, nil, LSTMScalar); out != nil || err == nil {
		t.Fatal("NaN input")
	}
	hs := append([]float32(nil), f.InitialHidden...)
	hs[0] = float32(math.Inf(1))
	if _, err := m.Forward(context.Background(), f.Input, f.Frames, hs, f.InitialCell, LSTMScalar); err == nil {
		t.Fatal("Inf state")
	}
	var nilModel *LSTM
	var zero LSTM
	for _, model := range []*LSTM{nilModel, &zero} {
		if out, err := model.Forward(context.Background(), nil, 1, nil, nil, LSTMScalar); err == nil || out != nil {
			t.Fatal("zero model")
		}
	}
}

func TestLSTMOverflowCancellationAndAllocationBound(t *testing.T) {
	f := loadLSTMFixtures(t)[1]
	count := newPowersetContext(0)
	m, err := NewLSTM(count, f.Config, f.Weights)
	count.cancel()
	if err != nil {
		t.Fatal(err)
	}
	for at := 1; at <= count.calls; at++ {
		ctx := newPowersetContext(at)
		out, err := NewLSTM(ctx, f.Config, f.Weights)
		ctx.cancel()
		if out != nil || !errors.Is(err, context.Canceled) {
			t.Fatal("constructor cancellation")
		}
	}
	for _, mode := range []LSTMMode{LSTMScalar, LSTMSIMD} {
		count := newPowersetContext(0)
		if _, err := m.Forward(count, f.Input, f.Frames, f.InitialHidden, f.InitialCell, mode); err != nil {
			t.Fatal(err)
		}
		count.cancel()
		t.Logf("forward cancellation checkpoints mode%d: %d", mode, count.calls)
		for at := 1; at <= count.calls; at++ {
			ctx := newPowersetContext(at)
			out, err := m.Forward(ctx, f.Input, f.Frames, f.InitialHidden, f.InitialCell, mode)
			ctx.cancel()
			if out != nil || !errors.Is(err, context.Canceled) {
				t.Fatal("forward cancellation")
			}
		}
	}
	// Cancel synchronously at an observed layer; later layers must not run.
	f = loadLSTMFixtures(t)[2]
	m, err = NewLSTM(context.Background(), f.Config, f.Weights)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	calls := 0
	out, err := m.ForwardObserved(ctx, f.Input, f.Frames, nil, nil, LSTMScalar, func(layer, _, _ int, _ []float32) {
		calls++
		if layer != 0 {
			t.Fatal("later layer ran")
		}
		cancel()
	})
	if out != nil || !errors.Is(err, context.Canceled) || calls != 1 {
		t.Fatal("observer cancellation")
	}
	// Overflowed affine arithmetic is rejected, not hidden by sigmoid saturation.
	small := loadLSTMFixtures(t)[0]
	small.Weights[0].Forward.WeightIH[0] = math.MaxFloat32
	m, err = NewLSTM(context.Background(), small.Config, small.Weights)
	if err != nil {
		t.Fatal(err)
	}
	for _, mode := range []LSTMMode{LSTMScalar, LSTMSIMD} {
		if out, err := m.Forward(context.Background(), []float32{math.MaxFloat32}, 1, nil, nil, mode); out != nil || err == nil {
			t.Fatal("overflow accepted")
		}
	}
	// Fixed allocations per call, not per recurrence step. No speed claim.
	f = loadLSTMFixtures(t)[2]
	m, err = NewLSTM(context.Background(), f.Config, f.Weights)
	if err != nil {
		t.Fatal(err)
	}
	var lastErr error
	var retained *LSTMResult
	short := testing.AllocsPerRun(5, func() {
		retained, lastErr = m.Forward(context.Background(), f.Input[:f.Config.InputSize], 1, nil, nil, LSTMSIMD)
	})
	long := testing.AllocsPerRun(5, func() { retained, lastErr = m.Forward(context.Background(), f.Input, f.Frames, nil, nil, LSTMSIMD) })
	if lastErr != nil || retained == nil || short != long {
		t.Fatalf("per-frame allocations short%g long%g err%v", short, long, lastErr)
	}
	t.Logf("LSTM allocation count per call: %.0f", long)
}

// Exercise representative PyanNet constructor dimensions through the existing
// SIMD GEMV surface, separately from the smaller independent Torch fixtures.
// This is synthetic shape parity, NOT a checkpoint geometry or speed claim.
func TestLSTMRepresentativeShapeScalarSIMDParity(t *testing.T) {
	cfg := LSTMConfig{InputSize: 60, HiddenSize: 128, NumLayers: 2, Bidirectional: true}
	layers := make([]LSTMLayer, cfg.NumLayers)
	for layer := range layers {
		width := cfg.InputSize
		if layer > 0 {
			width = 2 * cfg.HiddenSize
		}
		makeWeights := func(seed int) LSTMWeights {
			weights := LSTMWeights{WeightIH: make([]float32, 4*cfg.HiddenSize*width), WeightHH: make([]float32, 4*cfg.HiddenSize*cfg.HiddenSize), BiasIH: make([]float32, 4*cfg.HiddenSize), BiasHH: make([]float32, 4*cfg.HiddenSize)}
			for j, values := range [][]float32{weights.WeightIH, weights.WeightHH, weights.BiasIH, weights.BiasHH} {
				for i := range values {
					values[i] = float32((i*13+j*5+seed)%31-15) / 1024
				}
			}
			return weights
		}
		layers[layer] = LSTMLayer{Forward: makeWeights(layer + 1), Reverse: makeWeights(layer + 17)}
	}
	model, err := NewLSTM(context.Background(), cfg, layers)
	if err != nil {
		t.Fatal(err)
	}
	input := make([]float32, 3*cfg.InputSize)
	for i := range input {
		input[i] = float32(i%17-8) / 64
	}
	scalar, err := model.Forward(context.Background(), input, 3, nil, nil, LSTMScalar)
	if err != nil {
		t.Fatal(err)
	}
	native, err := model.Forward(context.Background(), input, 3, nil, nil, LSTMSIMD)
	if err != nil {
		t.Fatal(err)
	}
	closeLSTM(t, native.Output, scalar.Output)
	closeLSTM(t, native.Hidden, scalar.Hidden)
	closeLSTM(t, native.Cell, scalar.Cell)
}
