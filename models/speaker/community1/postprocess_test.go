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

type postprocessOracle struct {
	Name, Path                string
	Config                    PostprocessConfig
	Segmentations, Embeddings []*float32
	Counts                    []int
	OutputFrames              int   `json:"output_frames"`
	TrainingRows              int   `json:"training_rows"`
	TrainingChunks            []int `json:"training_chunks"`
	TrainingSpeakers          []int `json:"training_speakers"`
	Clusters, Classes         int
	Centroids, Scores         []float64
	Labels                    []int
	Full, Exclusive           []uint8
	FullTurns                 []SpeakerTurn `json:"full_turns"`
	ExclusiveTurns            []SpeakerTurn `json:"exclusive_turns"`
	ConstraintSatisfied       bool          `json:"constraint_satisfied"`
}

func loadPostprocessOracles(t *testing.T) []postprocessOracle {
	t.Helper()
	data, err := os.ReadFile("testdata/postprocess-reference.json")
	if err != nil {
		t.Fatal(err)
	}
	var f struct {
		Schema int
		Hashes map[string]string `json:"source_hashes"`
		Cases  []postprocessOracle
	}
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	if f.Schema != 1 || len(f.Cases) != 9 || f.Hashes["clustering"] != "6031fb7c21277a7e9901ef2cdaed7d5cd69f7ef45508dc4b45e82ce0da3c8fba" || f.Hashes["vbx"] != "a8c644feea4b381f9c1e7da72e0e47775c1fd482067e686801ddc16e5cac3c0e" {
		t.Fatal("postprocess oracle contract")
	}
	return f.Cases
}
func postprocessToyPLDA(t *testing.T) *PreparedPLDA {
	t.Helper()
	matrix := make([]float64, 16)
	for i := 0; i < 4; i++ {
		matrix[i*4+i] = 1
	}
	model, err := NewPreparedPLDA(context.Background(), PLDAConfig{4, 4, 4}, PreparedPLDAWeights{make([]float64, 4), make([]float64, 4), matrix, make([]float64, 4), matrix, []float64{.8, .6, .4, .2}})
	if err != nil {
		t.Fatal(err)
	}
	return model
}
func checkPostprocess(t *testing.T, out *PostprocessResult, c postprocessOracle) {
	t.Helper()
	if out.Path != c.Path || out.TrainingRows != c.TrainingRows || out.Clusters != c.Clusters || out.ConstraintSatisfied != c.ConstraintSatisfied || !reflect.DeepEqual(out.HardLabels, c.Labels) {
		t.Fatalf("postprocess %s status/labels got%+v want%v", c.Name, out, c.Labels)
	}
	closeClustering64(t, out.Centroids, c.Centroids)
	closeClustering64(t, out.SoftScores, c.Scores)
	if len(out.TrainingChunks) != len(c.TrainingChunks) || len(out.TrainingSpeakers) != len(c.TrainingSpeakers) {
		t.Fatal("training index count")
	}
	for i, v := range c.TrainingChunks {
		if out.TrainingChunks[i] != v || out.TrainingSpeakers[i] != c.TrainingSpeakers[i] {
			t.Fatal("training indices")
		}
	}
	timeline := out.Timeline
	if timeline.Frames != c.OutputFrames || timeline.Classes != c.Classes || !reflect.DeepEqual(timeline.Counts, c.Counts) || !reflect.DeepEqual(timeline.Full, c.Full) || !reflect.DeepEqual(timeline.Exclusive, c.Exclusive) {
		t.Fatal("postprocess timeline", c.Name, timeline.Counts, c.Counts)
	}
	if !reflect.DeepEqual(out.FullTurns, c.FullTurns) || !reflect.DeepEqual(out.ExclusiveTurns, c.ExclusiveTurns) {
		t.Fatal("postprocess turns", c.Name, out.FullTurns, c.FullTurns, out.ExclusiveTurns, c.ExclusiveTurns)
	}
}
func TestCommunity1PostprocessPinnedOracle(t *testing.T) {
	model := postprocessToyPLDA(t)
	for _, c := range loadPostprocessOracles(t) {
		t.Run(c.Name, func(t *testing.T) {
			seg, emb := maskValues(c.Segmentations), maskValues(c.Embeddings)
			beforeS, beforeE := append([]float32(nil), seg...), append([]float32(nil), emb...)
			var stages []string
			out, err := PostprocessCommunity1Observed(context.Background(), seg, emb, model, c.Config, func(stage string) error { stages = append(stages, stage); return nil })
			if err != nil {
				t.Fatal(err)
			}
			checkPostprocess(t, out, c)
			if !sameMaskBits(seg, beforeS) || !sameMaskBits(emb, beforeE) {
				t.Fatal("postprocess mutated source")
			}
			want := []string{"count", "filter", "ahc", "plda", "vbx", "centroids", "assignment", "reconstruction", "turns"}
			if c.Path == "silence" {
				want = []string{"count"}
			} else if c.Path == "single-training-row" {
				want = []string{"count", "filter", "centroids", "assignment", "reconstruction", "turns"}
			}
			if !reflect.DeepEqual(stages, want) {
				t.Fatal("stage order", stages, want)
			}
			for i := range seg {
				seg[i] = 0
			}
			for i := range emb {
				emb[i] = 999
			}
			checkPostprocess(t, out, c)
			if c.Path != "clustered" {
				again, err := PostprocessCommunity1(context.Background(), beforeS, beforeE, nil, c.Config)
				if err != nil {
					t.Fatal("sparse path required unused model", err)
				}
				checkPostprocess(t, again, c)
			}
		})
	}
}
func TestCommunity1PostprocessSparseAndCountPolicies(t *testing.T) {
	cases := loadPostprocessOracles(t)
	c := cases[0]
	model := postprocessToyPLDA(t)
	seg, emb := maskValues(c.Segmentations), maskValues(c.Embeddings)
	cfg := c.Config
	cfg.NumSpeakers = 3
	if out, err := PostprocessCommunity1(context.Background(), seg, emb, model, cfg); out != nil || !errors.Is(err, ErrSpeakerCountFallbackRequired) {
		t.Fatal("forced count silently substituted", err)
	}
	cfg.NumSpeakers = 2
	cfg.MinSpeakers = 4
	cfg.Reconstruction.MaxSpeakers = 1 // explicit Num overrides both
	if out, err := PostprocessCommunity1(context.Background(), seg, emb, model, cfg); err != nil || out.Clusters != 2 || !out.ConstraintSatisfied {
		t.Fatal("explicit count override", err)
	}
	for i := range seg {
		seg[i] = 1
	} // overlap-only active windows: no clean training
	if out, err := PostprocessCommunity1(context.Background(), seg, emb, model, c.Config); out != nil || !errors.Is(err, ErrNoTrainingEmbeddings) {
		t.Fatal("empty training manufactured centroid", err)
	}
	silence := cases[5]
	out, err := PostprocessCommunity1(context.Background(), maskValues(silence.Segmentations), nil, nil, silence.Config)
	if err != nil {
		t.Fatal(err)
	}
	checkPostprocess(t, out, silence)
	// Count-based silence exits even with a single raw active frame rounded to0.
	rounded := cases[6]
	out, err = PostprocessCommunity1(context.Background(), maskValues(rounded.Segmentations), nil, nil, rounded.Config)
	if err != nil {
		t.Fatal(err)
	}
	checkPostprocess(t, out, rounded)
}
func TestCommunity1PostprocessValidation(t *testing.T) {
	c := loadPostprocessOracles(t)[0]
	model := postprocessToyPLDA(t)
	for _, kind := range []string{"dimension", "minmax", "num", "max", "threshold", "fa", "fb", "minoff", "frames", "localspeakers", "step", "seglen", "soft", "infinite_seg", "emblen", "infinite_emb", "nan_emb", "zero_emb", "nilmodel", "zeromodel", "ties"} {
		cfg := c.Config
		seg, emb := maskValues(c.Segmentations), maskValues(c.Embeddings)
		plda := model
		switch kind {
		case "dimension":
			cfg.EmbeddingDimension = 513
		case "minmax":
			cfg.MinSpeakers = 5
		case "num":
			cfg.NumSpeakers = 65
		case "max":
			cfg.Reconstruction.MaxSpeakers = 0
		case "threshold":
			cfg.AHCThreshold = math.NaN()
		case "fa":
			cfg.Fa = 0
		case "fb":
			cfg.Fb = math.Inf(1)
		case "minoff":
			cfg.MinDurationOff = 31
		case "frames":
			cfg.Reconstruction.Frames = int(^uint(0) >> 1)
		case "localspeakers":
			cfg.Reconstruction.Speakers = 9
		case "step":
			cfg.Reconstruction.FrameStep = 0
		case "seglen":
			seg = seg[:len(seg)-1]
		case "soft":
			seg[0] = .3
		case "infinite_seg":
			seg[0] = float32(math.Inf(1))
		case "emblen":
			emb = emb[:len(emb)-1]
		case "infinite_emb":
			emb[0] = float32(math.Inf(1))
		case "nan_emb":
			emb[0] = float32(math.NaN())
		case "zero_emb":
			clear(emb[:4])
		case "nilmodel":
			plda = nil
		case "zeromodel":
			plda = &PreparedPLDA{}
		case "ties":
			overlap := loadPostprocessOracles(t)[2]
			cfg = overlap.Config
			seg = maskValues(overlap.Segmentations)
			emb = maskValues(overlap.Embeddings)
			cfg.Reconstruction.TiePolicy = RejectAmbiguousTies
		}
		if out, err := PostprocessCommunity1(context.Background(), seg, emb, plda, cfg); out != nil || err == nil {
			t.Fatal("accepted bad postprocess", kind)
		}
	}
}
func TestCommunity1PostprocessCapacityAndSparseCancellation(t *testing.T) {
	base := loadPostprocessOracles(t)[0].Config
	base.Reconstruction.Chunks = 513
	base.Reconstruction.Frames = 1
	base.Reconstruction.Speakers = 1
	base.Reconstruction.ChunkDuration = .125
	base.Reconstruction.ChunkStep = .125
	seg := make([]float32, 513)
	emb := make([]float32, 513*4)
	for i := range seg {
		seg[i] = 1
		emb[i*4] = 1
	}
	var stages []string
	out, err := PostprocessCommunity1Observed(context.Background(), seg, emb, postprocessToyPLDA(t), base, func(stage string) error { stages = append(stages, stage); return nil })
	if err == nil || out != nil || !reflect.DeepEqual(stages, []string{"count", "filter"}) {
		t.Fatal("AHC row cap not enforced before clustering", stages, err)
	}
	base.Reconstruction.Chunks = 65
	seg = seg[:65]
	emb = emb[:65*4]
	base.AHCThreshold = 0
	for i := range seg {
		angle := float64(i) * 2 * math.Pi / 65
		emb[i*4] = float32(math.Cos(angle))
		emb[i*4+1] = float32(math.Sin(angle))
	}
	if out, err := PostprocessCommunity1(context.Background(), seg, emb, postprocessToyPLDA(t), base); err == nil || out != nil {
		t.Fatal("initial VBx slot cap")
	}
	for _, index := range []int{3, 5, 6} {
		c := loadPostprocessOracles(t)[index]
		seg, emb := maskValues(c.Segmentations), maskValues(c.Embeddings)
		if index == 3 {
			cause := errors.New("single centroid observer error")
			out, err := PostprocessCommunity1Observed(context.Background(), seg, emb, nil, c.Config, func(stage string) error {
				if stage == "centroids" {
					return cause
				}
				return nil
			})
			if out != nil || !errors.Is(err, cause) {
				t.Fatal("single centroid error swallowed", err)
			}
		}
		count := newPowersetContext(0)
		_, err := PostprocessCommunity1(count, seg, emb, nil, c.Config)
		count.cancel()
		if err != nil {
			t.Fatal(err)
		}
		for at := 1; at <= count.calls; at++ {
			ctx := newPowersetContext(at)
			out, err := PostprocessCommunity1(ctx, seg, emb, nil, c.Config)
			ctx.cancel()
			if out != nil || !errors.Is(err, context.Canceled) {
				t.Fatal("sparse cancellation", index, at, err)
			}
		}
		t.Logf("sparse case%d checkpoints%d", index, count.calls)
	}
}

func TestCommunity1PostprocessCancellationObservers(t *testing.T) {
	c := loadPostprocessOracles(t)[0]
	model := postprocessToyPLDA(t)
	seg, emb := maskValues(c.Segmentations), maskValues(c.Embeddings)
	stages := []string{"count", "filter", "ahc", "plda", "vbx", "centroids", "assignment", "reconstruction", "turns"}
	cause := errors.New("observer rejected")
	for _, stage := range stages {
		out, err := PostprocessCommunity1Observed(context.Background(), seg, emb, model, c.Config, func(current string) error {
			if current == stage {
				return cause
			}
			return nil
		})
		if out != nil || !errors.Is(err, cause) {
			t.Fatal("observer error", stage, err)
		}
		ctx, cancel := context.WithCancel(context.Background())
		out, err = PostprocessCommunity1Observed(ctx, seg, emb, model, c.Config, func(current string) error {
			if current == stage {
				cancel()
			}
			return nil
		})
		cancel()
		if out != nil || !errors.Is(err, context.Canceled) {
			t.Fatal("observer cancellation", stage, err)
		}
	}
	counter := newPowersetContext(0)
	_, err := PostprocessCommunity1(counter, seg, emb, model, c.Config)
	counter.cancel()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("postprocess checkpoints: %d", counter.calls)
	for at := 1; at <= counter.calls; at++ {
		ctx := newPowersetContext(at)
		out, err := PostprocessCommunity1(ctx, seg, emb, model, c.Config)
		ctx.cancel()
		if out != nil || !errors.Is(err, context.Canceled) {
			t.Fatal("postprocess cancellation", at, err)
		}
	}
}
func TestCommunity1PostprocessConcurrencyAndReuse(t *testing.T) {
	c := loadPostprocessOracles(t)[0]
	model := postprocessToyPLDA(t)
	seg, emb := maskValues(c.Segmentations), maskValues(c.Embeddings)
	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			out, err := PostprocessCommunity1(context.Background(), seg, emb, model, c.Config)
			if err != nil {
				t.Error(err)
				return
			}
			checkPostprocess(t, out, c)
		}()
	}
	wg.Wait()
	out, err := PostprocessCommunity1(context.Background(), seg, emb, model, c.Config)
	if err != nil {
		t.Fatal(err)
	}
	for _, values := range [][]float64{out.Centroids, out.SoftScores} {
		for i := range values {
			values[i] = 99
		}
	}
	for i := range out.HardLabels {
		out.HardLabels[i] = -2
	}
	clear(out.Timeline.Full)
	clear(out.Timeline.Exclusive)
	again, err := PostprocessCommunity1(context.Background(), seg, emb, model, c.Config)
	if err != nil {
		t.Fatal(err)
	}
	checkPostprocess(t, again, c)
}
