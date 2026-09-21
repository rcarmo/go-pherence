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

type reconstructionOracle struct {
	Name            string
	Config          ReconstructionConfig
	Segmentations   []*float32
	Labels          []int
	OutputFrames    int `json:"output_frames"`
	Classes         int
	Counts          []int
	Scores          []float32
	Full, Exclusive []uint8
	StableFull      []uint8 `json:"stable_full"`
	StableExclusive []uint8 `json:"stable_exclusive"`
	Ambiguous       []int   `json:"ambiguous_frames"`
}
type turnsOracle struct {
	Name     string
	Config   BinaryTurnConfig
	Activity []uint8
	Turns    []SpeakerTurn
}

func loadReconstructionOracles(t *testing.T) ([]reconstructionOracle, []turnsOracle) {
	t.Helper()
	data, err := os.ReadFile("testdata/reconstruct-reference.json")
	if err != nil {
		t.Fatal(err)
	}
	var f struct {
		Schema int
		Hashes map[string]string `json:"source_hashes"`
		Core   map[string]string `json:"core_hashes"`
		Cases  []reconstructionOracle
		Turns  []turnsOracle
	}
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	if f.Schema != 1 || len(f.Cases) != 18 || len(f.Turns) != 18 || f.Hashes["inference"] != "c29f525c93a1a5c6bd0a9475ae4fb8c73ce9b48cf9735301302eb99886ad7d41" || f.Hashes["mixin"] != "14740a7286606c208db993d29fde5703d8323205959b7e7658069f385c2a63a5" || f.Hashes["signal"] != "334e6b9ca1a6e472dbc9b1562e6cdeb27940d13bd6eef8d4636a692a6ac4ab16" {
		t.Fatal("reconstruction fixture contract")
	}
	if f.Core["segment"] != "3d462a735e2a0a29a861373fbe94fd57301025c91a1b4b50c44a5f5268ef169f" || f.Core["feature"] != "bbff5b817108e0fe4d1fac08c70a253341a6408878d08eaf926c09801c279c3e" {
		t.Fatal("core timing fixture changed")
	}
	return f.Cases, f.Turns
}
func checkReconstruction(t *testing.T, out *ActivityTimeline, c reconstructionOracle) {
	t.Helper()
	if out.Frames != c.OutputFrames || out.Classes != c.Classes || !reflect.DeepEqual(out.Counts, c.Counts) || !reflect.DeepEqual(out.Activations, c.Scores) || !reflect.DeepEqual(out.Full, c.StableFull) || !reflect.DeepEqual(out.Exclusive, c.StableExclusive) {
		t.Fatalf("reconstruction mismatch %s frames%d/%d classes%d/%d counts%v/%v scores%v/%v", c.Name, out.Frames, c.OutputFrames, out.Classes, c.Classes, out.Counts, c.Counts, out.Activations, c.Scores)
	}
	if len(out.AmbiguousFrames) != len(c.Ambiguous) {
		t.Fatal("ambiguity report", out.AmbiguousFrames, c.Ambiguous)
	}
	for i, v := range out.AmbiguousFrames {
		if v != c.Ambiguous[i] {
			t.Fatal("ambiguity index")
		}
	}
	for frame, count := range out.Counts {
		full, exclusive := 0, 0
		for k := 0; k < out.Classes; k++ {
			full += int(out.Full[frame*out.Classes+k])
			exclusive += int(out.Exclusive[frame*out.Classes+k])
		}
		if full != count || exclusive != min(count, 1) {
			t.Fatal("activity count invariant")
		}
	}
}
func TestReconstructPowersetPinnedOracle(t *testing.T) {
	cases, _ := loadReconstructionOracles(t)
	ambiguous, unambiguous := 0, 0
	for _, c := range cases {
		t.Run(c.Name, func(t *testing.T) {
			seg := maskValues(c.Segmentations)
			before := append([]float32(nil), seg...)
			labels := append([]int(nil), c.Labels...)
			out, err := ReconstructPowerset(context.Background(), seg, labels, c.Config)
			if err != nil {
				t.Fatal(err)
			}
			checkReconstruction(t, out, c)
			if !sameMaskBits(seg, before) || !reflect.DeepEqual(labels, c.Labels) {
				t.Fatal("source mutation")
			}
			strict := c.Config
			strict.TiePolicy = RejectAmbiguousTies
			checked, err := ReconstructPowerset(context.Background(), seg, labels, strict)
			if len(c.Ambiguous) > 0 {
				ambiguous++
				if err == nil || checked != nil {
					t.Fatal("silent upstream tie substitution")
				}
			} else {
				unambiguous++
				if err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(checked.Full, c.Full) || !reflect.DeepEqual(checked.Exclusive, c.Exclusive) {
					t.Fatal("unambiguous reference mismatch")
				}
			}
			for i := range seg {
				seg[i] = 0
			}
			for i := range labels {
				labels[i] = -2
			}
			checkReconstruction(t, out, c)
		})
	}
	if ambiguous != 12 || unambiguous != 6 {
		t.Fatal("tie coverage", ambiguous, unambiguous)
	}
}
func TestBinaryTurnsPinnedOracle(t *testing.T) {
	_, cases := loadReconstructionOracles(t)
	for _, c := range cases {
		t.Run(c.Name, func(t *testing.T) {
			before := append([]uint8(nil), c.Activity...)
			turns, err := BinaryActivityToTurns(context.Background(), c.Activity, c.Config)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(turns, c.Turns) || !reflect.DeepEqual(before, c.Activity) {
				t.Fatal("turn oracle mismatch", turns, c.Turns)
			}
		})
	}
}
func TestReconstructionRoundAndTurnEdges(t *testing.T) {
	cfg := ReconstructionConfig{Chunks: 2, Frames: 2, Speakers: 2, Start: 0, ChunkDuration: .25, ChunkStep: .125, FrameDuration: .25, FrameStep: .125, MaxSpeakers: 2, TiePolicy: LowestIndexTies}
	// At overlapping frame: 1/2 rounds to0 (not away from zero).
	out, err := ReconstructPowerset(context.Background(), []float32{1, 0, 1, 0, 0, 0, 0, 0}, []int{0, 1, 0, 1}, cfg)
	if err != nil || out.Counts[1] != 0 {
		t.Fatal("nearest/even count", err, out)
	}
	// At overlapping frame: 3/2 rounds to2.
	out, err = ReconstructPowerset(context.Background(), []float32{1, 1, 1, 1, 1, 0, 1, 0}, []int{0, 1, 0, 1}, cfg)
	if err != nil || out.Counts[1] != 2 {
		t.Fatal("nearest/even count up", err, out)
	}
	// Final active frame is zero duration; no invented end extension.
	tc := BinaryTurnConfig{Frames: 3, Speakers: 1, Start: 0, FrameDuration: .25, FrameStep: .125}
	turns, err := BinaryActivityToTurns(context.Background(), []uint8{0, 0, 1}, tc)
	if err != nil || len(turns) != 0 {
		t.Fatal("last-frame tail extension")
	}
	tc.MinDurationOn = .251
	turns, err = BinaryActivityToTurns(context.Background(), []uint8{1, 1, 1}, tc)
	if err != nil || len(turns) != 0 {
		t.Fatal("min-on")
	}
	tc.MinDurationOn = .25
	turns, err = BinaryActivityToTurns(context.Background(), []uint8{1, 1, 1}, tc)
	if err != nil || len(turns) != 1 {
		t.Fatal("min-on equality")
	}
	// Reconstruct->full/exclusive->turns, still synthetic grid, no neural wiring.
	cfg = ReconstructionConfig{Chunks: 1, Frames: 4, Speakers: 1, Start: .5, ChunkDuration: .5, ChunkStep: .5, FrameDuration: .25, FrameStep: .125, MaxSpeakers: 1}
	out, err = ReconstructPowerset(context.Background(), []float32{1, 1, 0, 0}, []int{0}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	tc = BinaryTurnConfig{Frames: out.Frames, Speakers: out.Classes, Start: out.Start, FrameDuration: out.FrameDuration, FrameStep: out.FrameStep}
	full, err := BinaryActivityToTurns(context.Background(), out.Full, tc)
	if err != nil {
		t.Fatal(err)
	}
	exclusive, err := BinaryActivityToTurns(context.Background(), out.Exclusive, tc)
	if err != nil || !reflect.DeepEqual(full, []SpeakerTurn{{.625, .875, 0}}) || !reflect.DeepEqual(full, exclusive) {
		t.Fatal("reconstruction turn bridge", full, exclusive, err)
	}
}
func TestReconstructionValidation(t *testing.T) {
	cases, _ := loadReconstructionOracles(t)
	base := cases[6]
	for _, kind := range []string{"chunks", "frames", "speakers", "duration", "step", "start", "nan", "inf", "output_bound", "grid_extent", "maxspeakers", "tiepolicy", "short", "soft", "inf_seg", "label"} {
		c := base.Config
		seg := maskValues(base.Segmentations)
		labels := append([]int(nil), base.Labels...)
		switch kind {
		case "chunks":
			c.Chunks = int(^uint(0) >> 1)
		case "frames":
			c.Frames = 0
		case "speakers":
			c.Speakers = 9
		case "duration":
			c.ChunkDuration = 0
		case "step":
			c.FrameStep = 2
		case "start":
			c.Start = -1
		case "nan":
			c.Start = math.NaN()
		case "inf":
			c.ChunkStep = math.Inf(1)
		case "output_bound":
			c.FrameStep = 1e-6
			c.ChunkDuration = 30
		case "grid_extent":
			c.Frames = 4096
		case "maxspeakers":
			c.MaxSpeakers = 0
		case "tiepolicy":
			c.TiePolicy = 3
		case "short":
			seg = seg[:len(seg)-1]
		case "soft":
			seg[0] = .5
		case "inf_seg":
			seg[0] = float32(math.Inf(1))
		case "label":
			labels[0] = -1
		}
		if out, err := ReconstructPowerset(context.Background(), seg, labels, c); err == nil || out != nil {
			t.Fatal("accepted reconstruction", kind)
		}
	}
	tc := BinaryTurnConfig{Frames: 3, Speakers: 1, Start: 0, FrameDuration: .25, FrameStep: .125}
	for _, kind := range []string{"oneframe", "speakers", "short", "nonbinary", "start", "nan", "step", "minon", "minoff"} {
		c := tc
		x := []uint8{1, 0, 0}
		switch kind {
		case "oneframe":
			c.Frames = 1
		case "speakers":
			c.Speakers = 65
		case "short":
			x = x[:2]
		case "nonbinary":
			x[0] = 2
		case "start":
			c.Start = -1
		case "nan":
			c.MinDurationOff = math.NaN()
		case "step":
			c.FrameStep = 0
		case "minon":
			c.MinDurationOn = -1
		case "minoff":
			c.MinDurationOff = 31
		}
		if out, err := BinaryActivityToTurns(context.Background(), x, c); err == nil || out != nil {
			t.Fatal("accepted turn config", kind)
		}
	}
}
func TestReconstructionOutputBounds(t *testing.T) {
	// Input small, sparse cluster label makes output width64: reject >2^24.
	cfg := ReconstructionConfig{Chunks: 1, Frames: 1, Speakers: 1, ChunkDuration: 30, ChunkStep: 1, FrameDuration: .0001, FrameStep: .0001, MaxSpeakers: 1}
	if out, err := ReconstructPowerset(context.Background(), []float32{1}, []int{63}, cfg); err == nil || out != nil {
		t.Fatal("output element bound")
	}
	cfg.FrameDuration, cfg.FrameStep = .0004, .0004 // fits element limit, exceeds sort-work bound
	if out, err := ReconstructPowerset(context.Background(), []float32{1}, []int{63}, cfg); err == nil || out != nil {
		t.Fatal("sort work bound")
	}
	// Alternating activity exceeds the explicit turn bound before final sort.
	activity := make([]uint8, 200004)
	for i := 0; i < len(activity); i += 2 {
		activity[i] = 1
	}
	tc := BinaryTurnConfig{Frames: len(activity), Speakers: 1, FrameDuration: .001, FrameStep: .001}
	if out, err := BinaryActivityToTurns(context.Background(), activity, tc); err == nil || out != nil {
		t.Fatal("turn output bound")
	}
}

func TestReconstructionCancellationConcurrency(t *testing.T) {
	cases, turnCases := loadReconstructionOracles(t)
	c, tc := cases[15], turnCases[14]
	seg := maskValues(c.Segmentations)
	runs := []func(context.Context) error{
		func(ctx context.Context) error {
			out, err := ReconstructPowerset(ctx, seg, c.Labels, c.Config)
			if err != nil && out != nil {
				t.Fatal("partial reconstruction")
			}
			return err
		},
		func(ctx context.Context) error {
			out, err := BinaryActivityToTurns(ctx, tc.Activity, tc.Config)
			if err != nil && out != nil {
				t.Fatal("partial turns")
			}
			return err
		},
	}
	for stage, run := range runs {
		counter := newPowersetContext(0)
		err := run(counter)
		counter.cancel()
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("reconstruction stage%d checkpoints%d", stage, counter.calls)
		for at := 1; at <= counter.calls; at++ {
			ctx := newPowersetContext(at)
			err := run(ctx)
			ctx.cancel()
			if !errors.Is(err, context.Canceled) {
				t.Fatal("cancel", stage, at, err)
			}
		}
	}
	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			out, err := ReconstructPowerset(context.Background(), seg, c.Labels, c.Config)
			if err != nil {
				t.Error(err)
				return
			}
			checkReconstruction(t, out, c)
			turns, err := BinaryActivityToTurns(context.Background(), tc.Activity, tc.Config)
			if err != nil || !reflect.DeepEqual(turns, tc.Turns) {
				t.Error("concurrent turns", err)
			}
		}()
	}
	wg.Wait()
}
