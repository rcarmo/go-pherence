package audio

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
	"testing"
)

type wespeakerCase struct {
	Name                                string
	Input, Window, Power, Output, Means []float32
	LogMel                              []float32 `json:"log_mel"`
	Frames                              int
}
type wespeakerFixture struct {
	Schema    int
	Reference struct {
		Kaldi     string `json:"kaldi_sha256"`
		WeSpeaker string `json:"wespeaker_sha256"`
	}
	Hamming, Mel []float32
	Cases        []wespeakerCase
}

func loadWeSpeakerFixture(t *testing.T) wespeakerFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/wespeaker-fbank-reference.json.gz")
	if err != nil {
		t.Fatal(err)
	}
	z, err := gzip.NewReader(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	defer z.Close()
	data, err = io.ReadAll(io.LimitReader(z, 8<<20))
	if err != nil || len(data) >= 8<<20 {
		t.Fatal("invalid fixture", err)
	}
	var f wespeakerFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	if f.Schema != 1 || len(f.Cases) != 8 || f.Reference.Kaldi != "5cbea1a584ddea748f6f68a621d794e13334e88d9faa1d40986f7af32f196d29" || f.Reference.WeSpeaker != "a2c13a792c50d97f7a7583b69fcabd5eb5efc34d1374c2b6e5757bbcf3b62139" {
		t.Fatal("WeSpeaker reference changed")
	}
	return f
}
func compareWeSpeaker(t *testing.T, name string, got, want []float32, abs, rel float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatal("shape", name, len(got), len(want))
	}
	worst := 0.0
	for i, value := range got {
		difference := math.Abs(float64(value) - float64(want[i]))
		worst = math.Max(worst, difference)
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || difference > abs+rel*math.Abs(float64(want[i])) {
			t.Fatalf("%s[%d] got%.9g want%.9g difference%g bounds%g+%g*abs(ref)", name, i, value, want[i], difference, abs, rel)
		}
	}
	t.Logf("%s max abs %g", name, worst)
}

func TestWeSpeakerFbankPinnedOracle(t *testing.T) {
	f := loadWeSpeakerFixture(t)
	weSpeakerTables.Do(initWeSpeakerTables)
	compareWeSpeaker(t, "Hamming", weSpeakerTables.window[:], f.Hamming, 2e-7, 0)
	compareWeSpeaker(t, "mel triangles", weSpeakerTables.mel[:], f.Mel, 2e-5, 0)
	for _, c := range f.Cases {
		t.Run(c.Name, func(t *testing.T) {
			windows, powers, global := 0, 0, 0
			out, frames, err := WeSpeakerFbankObserved(context.Background(), c.Input, func(stage string, frame int, values []float32) {
				switch stage {
				case "window":
					if frame != windows {
						t.Fatal("window order")
					}
					windows++
					compareWeSpeaker(t, stage, values, c.Window[frame*512:(frame+1)*512], 0.002, 2e-5)
				case "power":
					powers++
					compareWeSpeaker(t, stage, values, c.Power[frame*257:(frame+1)*257], 0.1, 0.002)
				case "log_mel":
					global++
					compareWeSpeaker(t, stage, values, c.LogMel, 2e-4, 0)
				case "centered":
					global++
					compareWeSpeaker(t, stage, values, c.Output, 2e-4, 0)
				default:
					t.Fatal("unknown observer stage")
				}
			})
			if err != nil {
				t.Fatal(err)
			}
			if frames != c.Frames || windows != frames || powers != frames || global != 2 {
				t.Fatal("frame counts")
			}
			compareWeSpeaker(t, "returned", out, c.Output, 2e-4, 0)
		})
	}
}

func TestWeSpeakerFbankFrameCoverageAndMean(t *testing.T) {
	for _, n := range []int{400, 559, 560, 719, 720, 1001} {
		x := make([]float32, n)
		for i := range x {
			x[i] = float32((i*13)%31-15) / 64
		}
		out, frames, err := WeSpeakerFbank(context.Background(), x)
		if err != nil {
			t.Fatal(err)
		}
		if frames != 1+(n-400)/160 || len(out) != 80*frames {
			t.Fatal("off-by-one frames")
		}
		for band := 0; band < 80; band++ {
			mean := float64(0)
			for frame := 0; frame < frames; frame++ {
				mean += float64(out[frame*80+band])
			}
			if math.Abs(mean/float64(frames)) > 3e-6 {
				t.Fatal("uncentered band")
			}
		}
	}
	// Samples past the last complete frame must not affect output.
	x := make([]float32, 559)
	for i := range x {
		x[i] = float32(i%7) / 32
	}
	a, _, err := WeSpeakerFbank(context.Background(), x)
	if err != nil {
		t.Fatal(err)
	}
	for i := 400; i < len(x); i++ {
		x[i] = 999
	}
	b, _, err := WeSpeakerFbank(context.Background(), x)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(a, b) {
		t.Fatal("snipped tail leaked")
	}
}

func TestWeSpeakerFbankBoundsOwnershipAndCancellation(t *testing.T) {
	for _, n := range []int{0, 1, 399, 160001} {
		if out, frames, err := WeSpeakerFbank(context.Background(), make([]float32, n)); out != nil || frames != 0 || err == nil {
			t.Fatal("invalid length")
		}
	}
	for _, value := range []float32{float32(math.NaN()), float32(math.Inf(1)), math.MaxFloat32} {
		x := make([]float32, 400)
		x[0] = value
		if out, _, err := WeSpeakerFbank(context.Background(), x); out != nil || err == nil {
			t.Fatal("invalid/overflow sample")
		}
	}
	c := loadWeSpeakerFixture(t).Cases[3]
	original := append([]float32(nil), c.Input...)
	count := newMelCheckpointContext(0)
	out, _, err := WeSpeakerFbank(count, c.Input)
	count.cancel()
	if err != nil {
		t.Fatal(err)
	}
	for at := 1; at <= count.calls; at++ {
		ctx := newMelCheckpointContext(at)
		result, frames, err := WeSpeakerFbank(ctx, c.Input)
		ctx.cancel()
		if result != nil || frames != 0 || !errors.Is(err, context.Canceled) {
			t.Fatal("partial/cancel", at, err)
		}
	}
	t.Logf("WeSpeaker cancellation checkpoints: %d", count.calls)
	if !reflect.DeepEqual(original, c.Input) {
		t.Fatal("mutated PCM")
	}
	out[0] = 100
	b, _, err := WeSpeakerFbank(context.Background(), c.Input)
	if err != nil {
		t.Fatal(err)
	}
	compareWeSpeaker(t, "owned output", b, c.Output, 2e-4, 0)
	for _, target := range []string{"window", "power", "log_mel", "centered"} {
		ctx, cancel := context.WithCancel(context.Background())
		seen := false
		output, frames, err := WeSpeakerFbankObserved(ctx, c.Input, func(stage string, _ int, _ []float32) {
			if seen {
				t.Fatal("observer after cancelled stage")
			}
			if stage == target {
				seen = true
				cancel()
			}
		})
		cancel()
		if output != nil || frames != 0 || !seen || !errors.Is(err, context.Canceled) {
			t.Fatal("observer cancellation", target, err)
		}
	}
}

func TestWeSpeakerFbankPinnedMeanReduction(t *testing.T) {
	fixture := loadWeSpeakerFixture(t)
	var frame [400]float32
	for _, c := range fixture.Cases {
		if len(c.Means) != c.Frames {
			t.Fatal("missing reference frame means")
		}
		for index, want := range c.Means {
			for i, value := range c.Input[index*160 : index*160+400] {
				frame[i] = value * 32768
			}
			got := weSpeakerMean400(frame[:])
			if math.Float32bits(got) != math.Float32bits(want) {
				t.Fatalf("pinned mean case%s frame%d got%.12g want%.12g", c.Name, index, got, want)
			}
		}
	}
	// Constant residual that float64 arithmetic would incorrectly erase relative
	// to the pinned Torch float32 oracle before log-floor processing.
	for i := range frame {
		frame[i] = float32(.2) * 32768
	}
	if got := weSpeakerMean400(frame[:]); got != float32(6553.6005859375) {
		t.Fatal("constant float32 reduction changed", got)
	}
}
