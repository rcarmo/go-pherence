package community1

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"reflect"
	"sync"
	"testing"
)

type headCase struct {
	Config            HeadConfig
	Frames            int
	Input, Hard, Soft []float32
	Layers            []HeadLinear
	Classifier        HeadLinear
	Intermediates     [][]float32
	LogProbabilities  []float32 `json:"log_probabilities"`
}
type headFixture struct {
	Schema    int
	Tolerance float64 `json:"absolute_tolerance"`
	Reference struct {
		ModelSHA string `json:"model_sha256"`
		LSTMSHA  string `json:"lstm_fixture_sha256"`
	}
	Cases       []headCase
	Composition struct {
		LSTMCaseIndex int `json:"lstm_case_index"`
		Head          headCase
	}
}

func loadHeadFixtures(t *testing.T) headFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/head-reference.json")
	if err != nil {
		t.Fatal(err)
	}
	var f headFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	if f.Schema != 1 || f.Tolerance != 2e-6 || len(f.Cases) != 4 || f.Reference.ModelSHA != "baef26b0d3bcb0027b179f9d472cc5795f77073fa8e5210bdd2ce53a267275e3" {
		t.Fatal("head reference contract changed")
	}
	data, err = os.ReadFile("testdata/lstm-reference.json")
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(data)
	if hex.EncodeToString(sum[:]) != f.Reference.LSTMSHA {
		t.Fatal("composition LSTM fixture changed")
	}
	return f
}
func checkHeadProbabilities(t *testing.T, logp []float32, classes int) {
	t.Helper()
	for start := 0; start < len(logp); start += classes {
		sum := float64(0)
		for _, value := range logp[start : start+classes] {
			if value > 0 || math.IsNaN(float64(value)) {
				t.Fatal("invalid log probability")
			}
			sum += math.Exp(float64(value))
		}
		if math.Abs(sum-1) > 2e-6 {
			t.Fatalf("probabilities not normalised: %.10g", sum)
		}
	}
}

func TestHeadPinnedTorchOracle(t *testing.T) {
	for index, c := range loadHeadFixtures(t).Cases {
		for _, mode := range []HeadMode{HeadScalar, HeadSIMD} {
			t.Run(fmt.Sprintf("case%d-mode%d", index, mode), func(t *testing.T) {
				h, err := NewSegmentationHead(context.Background(), c.Config, c.Layers, c.Classifier)
				if err != nil {
					t.Fatal(err)
				}
				observer := 0
				output, err := h.ForwardObserved(context.Background(), c.Input, c.Frames, mode, func(layer, frames, features int, values []float32) {
					if layer != observer || frames != c.Frames || features*frames != len(values) {
						t.Fatal("head boundary geometry/order")
					}
					closeLSTM(t, values, c.Intermediates[layer])
					observer++
				})
				if err != nil {
					t.Fatal(err)
				}
				if observer != c.Config.NumLayers+1 {
					t.Fatal("missing head boundaries")
				}
				closeLSTM(t, output, c.LogProbabilities)
				checkHeadProbabilities(t, output, h.Classes())
				p, err := NewPowerset(c.Config.Speakers, c.Config.MaxActive)
				if err != nil {
					t.Fatal(err)
				}
				hard, err := p.Decode(context.Background(), output, c.Frames, PowersetHard)
				if err != nil {
					t.Fatal(err)
				}
				closePowerset(t, hard, c.Hard, 0)
				soft, err := p.Decode(context.Background(), output, c.Frames, PowersetSoft)
				if err != nil {
					t.Fatal(err)
				}
				closeLSTM(t, soft, c.Soft)
			})
		}
	}
}

func TestHeadLSTMPowersetComposition(t *testing.T) {
	f := loadHeadFixtures(t)
	c := f.Composition.Head
	recurrent := loadLSTMFixtures(t)[f.Composition.LSTMCaseIndex]
	lstm, err := NewLSTM(context.Background(), recurrent.Config, recurrent.Weights)
	if err != nil {
		t.Fatal(err)
	}
	head, err := NewSegmentationHead(context.Background(), c.Config, c.Layers, c.Classifier)
	if err != nil {
		t.Fatal(err)
	}
	for _, mode := range []HeadMode{HeadScalar, HeadSIMD} {
		recurrenceMode := LSTMScalar
		if mode == HeadSIMD {
			recurrenceMode = LSTMSIMD
		}
		features, err := lstm.Forward(context.Background(), recurrent.Input, recurrent.Frames, nil, nil, recurrenceMode)
		if err != nil {
			t.Fatal(err)
		}
		closeLSTM(t, features.Output, c.Input)
		logp, err := head.Forward(context.Background(), features.Output, recurrent.Frames, mode)
		if err != nil {
			t.Fatal(err)
		}
		closeLSTM(t, logp, c.LogProbabilities)
		powerset, _ := NewPowerset(c.Config.Speakers, c.Config.MaxActive)
		hard, err := powerset.Decode(context.Background(), logp, recurrent.Frames, PowersetHard)
		if err != nil {
			t.Fatal(err)
		}
		closePowerset(t, hard, c.Hard, 0)
		soft, err := powerset.Decode(context.Background(), logp, recurrent.Frames, PowersetSoft)
		if err != nil {
			t.Fatal(err)
		}
		closeLSTM(t, soft, c.Soft)
	}
	// This composition intentionally starts at synthetic recurrent input, not PCM.
	// It cannot qualify SincNet, true speaker identities, timestamps or diarization.
}

func TestHeadOwnershipConcurrencyAndAllocationBound(t *testing.T) {
	c := loadHeadFixtures(t).Cases[2]
	h, err := NewSegmentationHead(context.Background(), c.Config, c.Layers, c.Classifier)
	if err != nil {
		t.Fatal(err)
	}
	for _, w := range append(c.Layers, c.Classifier) {
		for _, values := range [][]float32{w.Weight, w.Bias} {
			for i := range values {
				values[i] = 99
			}
		}
	}
	before := append([]float32(nil), c.Input...)
	out, err := h.Forward(context.Background(), c.Input, c.Frames, HeadSIMD)
	if err != nil {
		t.Fatal(err)
	}
	closeLSTM(t, out, c.LogProbabilities)
	out[0] = 100
	out, err = h.Forward(context.Background(), c.Input, c.Frames, HeadSIMD)
	if err != nil {
		t.Fatal(err)
	}
	closeLSTM(t, out, c.LogProbabilities)
	if !reflect.DeepEqual(before, c.Input) {
		t.Fatal("input mutated")
	}
	var wg sync.WaitGroup
	for i := 0; i < 3; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			out, err := h.Forward(context.Background(), c.Input, c.Frames, HeadSIMD)
			if err != nil {
				t.Error(err)
				return
			}
			closeLSTM(t, out, c.LogProbabilities)
		}()
	}
	wg.Wait()
	var kept []float32
	var lastErr error
	short := testing.AllocsPerRun(5, func() { kept, lastErr = h.Forward(context.Background(), c.Input[:c.Config.InputSize], 1, HeadSIMD) })
	long := testing.AllocsPerRun(5, func() { kept, lastErr = h.Forward(context.Background(), c.Input, c.Frames, HeadSIMD) })
	if lastErr != nil || kept == nil || short != long {
		t.Fatalf("per-frame allocation short%g long%g %v", short, long, lastErr)
	}
	t.Logf("head allocations per call: %.0f", long)
}

func TestHeadRejectsMalformedGeometryAndData(t *testing.T) {
	for _, cfg := range []HeadConfig{{}, {513, 1, 1, 3, 2}, {1, 0, 1, 3, 2}, {1, 513, 1, 3, 2}, {1, 1, 5, 3, 2}, {1, 1, -1, 3, 2}, {1, 1, 0, 3, 2}, {1, 0, 0, 0, 0}, {int(^uint(0) >> 1), 0, 0, 3, 2}} {
		if h, err := NewSegmentationHead(context.Background(), cfg, nil, HeadLinear{}); err == nil || h != nil {
			t.Fatal("invalid head geometry")
		}
	}
	for _, kind := range []string{"layers", "hiddenWeight", "hiddenBias", "classifierWeight", "classifierBias", "nan", "inf"} {
		c := loadHeadFixtures(t).Cases[2]
		switch kind {
		case "layers":
			c.Layers = nil
		case "hiddenWeight":
			c.Layers[0].Weight = nil
		case "hiddenBias":
			c.Layers[0].Bias = nil
		case "classifierWeight":
			c.Classifier.Weight = nil
		case "classifierBias":
			c.Classifier.Bias = nil
		case "nan":
			c.Layers[0].Weight[0] = float32(math.NaN())
		case "inf":
			c.Classifier.Bias[0] = float32(math.Inf(1))
		}
		if h, err := NewSegmentationHead(context.Background(), c.Config, c.Layers, c.Classifier); err == nil || h != nil {
			t.Fatal("accepted malformed head", kind)
		}
	}
	c := loadHeadFixtures(t).Cases[2]
	h, err := NewSegmentationHead(context.Background(), c.Config, c.Layers, c.Classifier)
	if err != nil {
		t.Fatal(err)
	}
	for _, frames := range []int{-1, 0, 4097, int(^uint(0) >> 1)} {
		if out, err := h.Forward(context.Background(), nil, frames, HeadScalar); out != nil || err == nil {
			t.Fatal("invalid frames")
		}
	}
	if out, err := h.Forward(context.Background(), c.Input[:len(c.Input)-1], c.Frames, HeadScalar); out != nil || err == nil {
		t.Fatal("invalid input length")
	}
	if _, err := h.Forward(context.Background(), c.Input, c.Frames, HeadMode(3)); err == nil {
		t.Fatal("invalid mode")
	}
	c.Input[0] = float32(math.NaN())
	if out, err := h.Forward(context.Background(), c.Input, c.Frames, HeadSIMD); out != nil || err == nil {
		t.Fatal("nonfinite input")
	}
	var nilHead *SegmentationHead
	var zero SegmentationHead
	for _, head := range []*SegmentationHead{nilHead, &zero} {
		if out, err := head.Forward(context.Background(), nil, 1, HeadScalar); out != nil || err == nil {
			t.Fatal("nil/zero head")
		}
	}
	if nilHead.Classes() != 0 {
		t.Fatal("nil metadata")
	}
}

func TestHeadStableLogSoftmaxExtremesAndActivation(t *testing.T) {
	for _, bias := range [][]float32{{0, 0}, {1e30, 1e30}, {math.MaxFloat32, -math.MaxFloat32}, {-10000, -10001}, {10000, 9999}} {
		h, err := NewSegmentationHead(context.Background(), HeadConfig{InputSize: 1, Speakers: 1, MaxActive: 1}, nil, HeadLinear{Weight: []float32{0, 0}, Bias: bias})
		if err != nil {
			t.Fatal(err)
		}
		for _, mode := range []HeadMode{HeadScalar, HeadSIMD} {
			out, err := h.Forward(context.Background(), []float32{0}, 1, mode)
			if err != nil {
				t.Fatal(err)
			}
			checkHeadProbabilities(t, out, 2)
			p, _ := NewPowerset(1, 1)
			if _, err := p.Decode(context.Background(), out, 1, PowersetSoft); err != nil {
				t.Fatal("head/powerset extreme mismatch", err)
			}
			if bias[0] == bias[1] {
				closeLSTM(t, out, []float32{float32(-math.Ln2), float32(-math.Ln2)})
			}
		}
	}
	// Independent arithmetic: hidden activation slope0.01, NO activation at
	// classifier, and logsoftmax rather than sigmoid or plain softmax.
	h, err := NewSegmentationHead(context.Background(), HeadConfig{InputSize: 1, HiddenSize: 1, NumLayers: 1, Speakers: 1, MaxActive: 1}, []HeadLinear{{Weight: []float32{1}, Bias: []float32{0}}}, HeadLinear{Weight: []float32{1, -1}, Bias: []float32{0, 0}})
	if err != nil {
		t.Fatal(err)
	}
	out, err := h.Forward(context.Background(), []float32{-2}, 1, HeadScalar)
	if err != nil {
		t.Fatal(err)
	}
	normalizer := math.Log(math.Exp(-0.02) + math.Exp(0.02))
	closeLSTM(t, out, []float32{float32(-0.02 - normalizer), float32(0.02 - normalizer)})
	overflow, err := NewSegmentationHead(context.Background(), HeadConfig{InputSize: 1, Speakers: 1, MaxActive: 1}, nil, HeadLinear{Weight: []float32{math.MaxFloat32, 1}, Bias: []float32{0, 0}})
	if err != nil {
		t.Fatal(err)
	}
	for _, mode := range []HeadMode{HeadScalar, HeadSIMD} {
		if out, err := overflow.Forward(context.Background(), []float32{math.MaxFloat32}, 1, mode); out != nil || err == nil {
			t.Fatal("affine overflow hidden")
		}
	}
}

func TestHeadCancellationAtEveryCheckpoint(t *testing.T) {
	c := loadHeadFixtures(t).Cases[2]
	count := newPowersetContext(0)
	h, err := NewSegmentationHead(count, c.Config, c.Layers, c.Classifier)
	count.cancel()
	if err != nil {
		t.Fatal(err)
	}
	for at := 1; at <= count.calls; at++ {
		ctx := newPowersetContext(at)
		head, err := NewSegmentationHead(ctx, c.Config, c.Layers, c.Classifier)
		ctx.cancel()
		if head != nil || !errors.Is(err, context.Canceled) {
			t.Fatal("constructor cancellation", at, err)
		}
	}
	for _, mode := range []HeadMode{HeadScalar, HeadSIMD} {
		count := newPowersetContext(0)
		if _, err := h.Forward(count, c.Input, c.Frames, mode); err != nil {
			t.Fatal(err)
		}
		count.cancel()
		t.Logf("head forward cancel checkpoints mode%d: %d", mode, count.calls)
		for at := 1; at <= count.calls; at++ {
			ctx := newPowersetContext(at)
			out, err := h.Forward(ctx, c.Input, c.Frames, mode)
			ctx.cancel()
			if out != nil || !errors.Is(err, context.Canceled) {
				t.Fatal("forward cancellation", at, err)
			}
		}
	}
	for target := 0; target <= c.Config.NumLayers; target++ {
		ctx, cancel := context.WithCancel(context.Background())
		observed := 0
		out, err := h.ForwardObserved(ctx, c.Input, c.Frames, HeadSIMD, func(layer, _, _ int, _ []float32) {
			observed++
			if layer == target {
				cancel()
			}
		})
		cancel()
		if out != nil || !errors.Is(err, context.Canceled) || observed != target+1 {
			t.Fatal("observer cancellation")
		}
	}
}
