//go:build linux && amd64

package speechjob

import (
	"bytes"
	"compress/gzip"
	"context"
	"encoding/json"
	"errors"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/rcarmo/go-pherence/loader/audio/media"
	"github.com/rcarmo/go-pherence/loader/safetensors"
	c1 "github.com/rcarmo/go-pherence/model/speaker/community1"
)

func communityConfig() Community1StageConfig {
	h := hash([]byte("fixture identity"))
	return Community1StageConfig{AllowExperimental: true, SegmentationSHA256: h, EmbeddingSHA256: h, PLDASHA256: h, FiltersSHA256: h, RuntimeSHA256: h, PCM: c1.DiarizationPCMConfig{WindowSamples: 2960, StepSamples: 1600, MinimumEmbeddingSamples: 400, ExcludeOverlap: true, MinSpeakers: 1, MaxSpeakers: 4, AHCThreshold: .6, Fa: .07, Fb: .8, Constrained: true, TiePolicy: c1.RejectAmbiguousTies}, SegmentationModes: c1.SegmentationModes{SincNet: c1.SincNetSIMDFMA, LSTM: c1.LSTMSIMD, Head: c1.HeadSIMD}, EmbeddingMode: c1.WeSpeakerBlockSIMD, MaxResultBytes: 65536}
}
func communityFixtureResult(total int64, cfg Community1StageConfig) *c1.DiarizationPCMResult {
	windows, _ := c1.PlanDiarizationWindows(total, cfg.PCM.WindowSamples, cfg.PCM.StepSamples)
	end := float64(cfg.PCM.WindowSamples)/16000 + float64(len(windows)-1)*(float64(cfg.PCM.StepSamples)/16000)
	duration, step := .02, .01
	frames := int(math.RoundToEven(((end+.5*duration)-.5*duration)/step)) + 1
	return &c1.DiarizationPCMResult{Windows: windows, Grid: c1.SincNetGrid{Frames: 10, Step: 160, FirstCenter: 160, ReceptiveField: 320}, LocalSpeakers: 3, EmbeddingDimension: 4, Postprocess: &c1.PostprocessResult{Path: "single-training-row", TrainingRows: 1, Clusters: 1, ConstraintSatisfied: true, Timeline: &c1.ActivityTimeline{Frames: frames, Classes: 1, FrameDuration: duration, FrameStep: step}, FullTurns: []c1.SpeakerTurn{{Start: .01, End: float64(frames-1)*step + .01, Speaker: 0}}, ExclusiveTurns: []c1.SpeakerTurn{{Start: .01, End: float64(frames-1)*step + .01, Speaker: 0}}}}
}
func TestCommunityStageRetentionRetryAndIdentity(t *testing.T) {
	s, dir := openTest(t)
	job := createTest(t, s)
	cfg := communityConfig()
	calls := 0
	infer := func(ctx context.Context, reader c1.DiarizationPCMReader, total int64) (*c1.DiarizationPCMResult, error) {
		calls++
		if calls == 1 {
			return nil, io.ErrUnexpectedEOF
		}
		return communityFixtureResult(total, cfg), nil
	}
	st := community1Stage(cfg, infer)
	stages := []Stage{fixturePCMStage(3361), textStage("asr-windows", "raw saved ASR"), stage("transcript", func(ctx context.Context, _ *Input, w io.Writer) error {
		return WriteTranscriptJSON(ctx, w, transcriptFixture())
	}), NewVTTStage(), st}
	job, e := s.Run(context.Background(), job.ID, config, stages, nil)
	if !errors.Is(e, io.ErrUnexpectedEOF) || job.Status != Failed || len(job.Checkpoints) != 4 {
		t.Fatal(job, e)
	}
	r, e := s.OpenCheckpoint(context.Background(), job.ID, "vtt")
	vtt := readAll(t, r, e)
	if !strings.Contains(vtt, "Olá") {
		t.Fatal(vtt)
	}
	if e = s.Close(); e != nil {
		t.Fatal(e)
	}
	s, e = Open(dir, limits())
	if e != nil {
		t.Fatal(e)
	}
	defer s.Close()
	job, e = s.Run(context.Background(), job.ID, config, stages, nil)
	if e != nil || job.Status != Complete || calls != 2 {
		t.Fatal(job, calls, e)
	}
	r, e = s.OpenCheckpoint(context.Background(), job.ID, "diarization")
	raw := readAll(t, r, e)
	doc, e := ReadDiarizationJSON(context.Background(), strings.NewReader(raw))
	if e != nil {
		t.Fatal(e)
	}
	if !doc.Experimental || doc.Policy.TiePolicy != c1.RejectAmbiguousTies || doc.FullTurns[0].End <= float64(doc.TotalSamples)/16000 {
		t.Fatal("raw padded timing changed", doc)
	}
	if _, e = s.Run(context.Background(), job.ID, config, stages, nil); e != nil || calls != 2 {
		t.Fatal("repeated complete stage", e)
	}
	cfg.PCM.TiePolicy = c1.LowestIndexTies
	stages[4] = community1Stage(cfg, infer)
	if _, e = s.Run(context.Background(), job.ID, config, stages, nil); !errors.Is(e, ErrConfiguration) {
		t.Fatal("tie policy identity reuse", e)
	}
	assertNoScratch(t, s, job.ID)
	if e = s.Delete(context.Background(), job.ID); e != nil {
		t.Fatal(e)
	}
	if _, e = os.Stat(filepath.Join(dir, job.ID)); !errors.Is(e, os.ErrNotExist) {
		t.Fatal(e)
	}
}
func TestCommunityStageAdmissionCancelAndFailures(t *testing.T) {
	for _, kind := range []string{"quota", "too-long", "nil", "model-error", "panic", "cancel", "output-cap", "invalid-result"} {
		t.Run(kind, func(t *testing.T) {
			s, _ := openTest(t)
			job := createTest(t, s)
			cfg := communityConfig()
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			calls := 0
			infer := func(ctx context.Context, _ c1.DiarizationPCMReader, total int64) (*c1.DiarizationPCMResult, error) {
				calls++
				switch kind {
				case "nil":
					return nil, nil
				case "model-error":
					return nil, c1.ErrNoTrainingEmbeddings
				case "panic":
					panic("fixture")
				case "cancel":
					cancel()
				}
				r := communityFixtureResult(total, cfg)
				if kind == "invalid-result" {
					r.Postprocess.Timeline.FrameStep = math.NaN()
				}
				return r, nil
			}
			samples := 3361
			if kind == "too-long" {
				samples = cfg.PCM.WindowSamples + 128*cfg.PCM.StepSamples
			}
			if kind == "quota" {
				s.limits.MaxBytes = 195000
			}
			if kind == "output-cap" {
				cfg.MaxResultBytes = 1
			}
			job, e := s.Run(ctx, job.ID, config, []Stage{fixturePCMStage(samples), textStage("asr-windows", "preserved"), community1Stage(cfg, infer)}, nil)
			if e == nil || job.Status == Complete || len(job.Checkpoints) != 2 {
				t.Fatal(kind, job, e)
			}
			if (kind == "quota" || kind == "too-long") && calls != 0 {
				t.Fatal("unadmitted inference")
			}
			if kind == "cancel" && !errors.Is(e, context.Canceled) {
				t.Fatal(e)
			}
			if kind == "model-error" && !errors.Is(e, c1.ErrNoTrainingEmbeddings) {
				t.Fatal(e)
			}
			r, e := s.OpenCheckpoint(context.Background(), job.ID, "asr-windows")
			if readAll(t, r, e) != "preserved" {
				t.Fatal("ASR lost")
			}
			assertNoScratch(t, s, job.ID)
		})
	}
}
func TestCommunityConstructorPolicy(t *testing.T) {
	for _, kind := range []string{"optin", "hash", "duration", "step", "minembed", "speaker", "nan", "tie", "sinc", "lstm", "head", "embed", "bytes"} {
		c := communityConfig()
		switch kind {
		case "optin":
			c.AllowExperimental = false
		case "hash":
			c.PLDASHA256 = ""
		case "duration":
			c.PCM.WindowSamples = 399
		case "step":
			c.PCM.StepSamples = 0
		case "minembed":
			c.PCM.MinimumEmbeddingSamples = 0
		case "speaker":
			c.PCM.MinSpeakers = 5
		case "nan":
			c.PCM.Fa = math.NaN()
		case "tie":
			c.PCM.TiePolicy = 99
		case "sinc":
			c.SegmentationModes.SincNet = c1.SincNetScalar
		case "lstm":
			c.SegmentationModes.LSTM = 99
		case "head":
			c.SegmentationModes.Head = 99
		case "embed":
			c.EmbeddingMode = 99
		case "bytes":
			c.MaxResultBytes = 17 << 20
		}
		if _, e := NewCommunity1Stage(&c1.ExperimentalDiarization{}, c); e == nil {
			t.Fatal("bad policy", kind)
		}
	}
	if _, e := NewCommunity1Stage(nil, communityConfig()); e == nil {
		t.Fatal("nil model")
	}
}
func TestCommunityStagePersistsSourceTiming(t *testing.T) {
	cfg := communityConfig()
	source := media.SourceTiming{Start: 125 * time.Millisecond, Duration: time.Second, HasEdits: true, SourceRate: 48000, Priming: 1024, Padding: 512}
	st := community1Stage(cfg, func(context.Context, c1.DiarizationPCMReader, int64) (*c1.DiarizationPCMResult, error) {
		return communityFixtureResult(3361, cfg), nil
	})
	s, _ := openTest(t)
	job := createTest(t, s)
	job, err := s.Run(context.Background(), job.ID, config, []Stage{timedPCMStage(3361, source), st}, nil)
	if err != nil || job.Status != Complete {
		t.Fatal(job, err)
	}
	r, err := s.OpenCheckpoint(context.Background(), job.ID, "diarization")
	if err != nil {
		t.Fatal(err)
	}
	doc, err := ReadDiarizationJSON(context.Background(), r)
	if closeErr := r.Close(); err == nil {
		err = closeErr
	}
	if err != nil || doc.SourceTiming != source || doc.TotalSamples != 3361 {
		t.Fatalf("source timing not preserved: %+v %v", doc.SourceTiming, err)
	}
}

func TestDiarizationDocumentRejectsAndRoundTrips(t *testing.T) {
	cfg := communityConfig()
	key := hash([]byte("key"))
	ctx := context.Background()
	good, e := diarizationDocument(ctx, communityFixtureResult(3361, cfg), cfg.PCM, 3361, key)
	if e != nil {
		t.Fatal(e)
	}
	raw, _ := json.Marshal(good)
	raw = append(raw, '\n')
	read, e := ReadDiarizationJSON(ctx, bytes.NewReader(raw))
	if e != nil || !reflect.DeepEqual(read, good) {
		t.Fatal(read, e)
	}
	kmeansResult := communityFixtureResult(3361, cfg)
	kmeansResult.Postprocess.Path = "clustered-kmeans"
	kmeansResult.Postprocess.TrainingRows = 3
	kmeansResult.Postprocess.Clusters = 3
	kmeansResult.Postprocess.ConstraintSatisfied = true
	kmeansResult.Postprocess.Timeline.Classes = 3
	kmeans, e := diarizationDocument(ctx, kmeansResult, cfg.PCM, 3361, key)
	if e != nil || kmeans.Path != "clustered-kmeans" {
		t.Fatal("forced-count document", kmeans, e)
	}
	encoded, e := json.Marshal(kmeans)
	if e != nil {
		t.Fatal(e)
	}
	decoded, e := ReadDiarizationJSON(ctx, bytes.NewReader(append(encoded, '\n')))
	if e != nil || !reflect.DeepEqual(decoded, kmeans) {
		t.Fatal("forced-count document round trip", decoded, e)
	}
	for _, kind := range []string{"schema", "experimental", "key", "windows", "grid", "frames", "classes", "path", "rows", "constraints", "ties", "nan", "order", "label", "extent", "speaker-overlap"} {
		r := communityFixtureResult(3361, cfg)
		d, e := diarizationDocument(ctx, r, cfg.PCM, 3361, key)
		if e != nil {
			t.Fatal(e)
		}
		switch kind {
		case "schema":
			d.Schema++
		case "experimental":
			d.Experimental = false
		case "key":
			d.StageKey = "bad"
		case "windows":
			d.Windows[0].Padding++
		case "grid":
			d.SegmentationGrid.Step++
		case "frames":
			d.Timeline.Frames++
		case "classes":
			d.Timeline.Classes = 0
		case "path":
			d.Path = "automatic-fallback"
		case "rows":
			d.TrainingRows = 0
		case "constraints":
			d.ConstraintSatisfied = false
		case "ties":
			d.AmbiguousFrames = []int{1}
		case "nan":
			d.FullTurns[0].Start = math.NaN()
		case "order":
			d.FullTurns = append(d.FullTurns, c1.SpeakerTurn{Start: 0, End: .02})
		case "label":
			d.FullTurns[0].Speaker = 1
		case "extent":
			d.FullTurns[0].End = 100
		case "speaker-overlap":
			d.FullTurns = append(d.FullTurns, c1.SpeakerTurn{Start: .02, End: .03})
		}
		if e = validateDiarizationDocument(ctx, d); e == nil {
			t.Fatal("invalid document", kind)
		}
	}
	for _, data := range [][]byte{[]byte("null"), append([]byte(" "), raw...), bytes.Replace(raw, []byte(`"schema":2`), []byte(`"schema":2,"schema":2`), 1), bytes.Replace(raw, []byte(`"schema":2`), []byte(`"Schema":2`), 1), bytes.Replace(raw, []byte(`"experimental":true,`), nil, 1), bytes.Repeat([]byte{'x'}, (16<<20)+1)} {
		if _, e = ReadDiarizationJSON(ctx, bytes.NewReader(data)); e == nil {
			t.Fatal("ambiguous/oversized JSON")
		}
	}
	cfg.PCM.TiePolicy = c1.LowestIndexTies
	cfg.PCM.NumSpeakers = 5
	r := communityFixtureResult(3361, cfg)
	r.Postprocess.ConstraintSatisfied = false
	r.Postprocess.Timeline.AmbiguousFrames = []int{1, 3}
	if _, e = diarizationDocument(ctx, r, cfg.PCM, 3361, key); e != nil {
		t.Fatal("explicit count/tie diagnostics lost", e)
	}
	r.Postprocess.Path = "silence"
	r.Postprocess.Clusters = 0
	r.Postprocess.TrainingRows = 0
	r.Postprocess.Timeline.Classes = 0
	r.Postprocess.FullTurns = nil
	r.Postprocess.ExclusiveTurns = nil
	r.Postprocess.Timeline.AmbiguousFrames = nil
	if _, e = diarizationDocument(ctx, r, cfg.PCM, 3361, key); e != nil {
		t.Fatal("silence", e)
	}
}

// Fixture loading uses only checked-in, reduced synthetic reference tensors.
// It does not open projects/models or any trained checkpoint/runtime.
type communityFixtureSource struct {
	infos  map[string]safetensors.TensorInfo
	values map[string][]float32
}

func (s *communityFixtureSource) TensorInfos() map[string]safetensors.TensorInfo { return s.infos }
func (s *communityFixtureSource) GetFloat32(name string) ([]float32, []int, error) {
	return s.values[name], s.infos[name].Shape, nil
}
func readCommunityFixture(t *testing.T, name string, out any) {
	t.Helper()
	f, e := os.Open("../../model/speaker/community1/testdata/" + name)
	if e != nil {
		t.Fatal(e)
	}
	defer f.Close()
	z, e := gzip.NewReader(f)
	if e != nil {
		t.Fatal(e)
	}
	defer z.Close()
	b, e := io.ReadAll(io.LimitReader(z, 16<<20))
	if e != nil || len(b) >= 16<<20 {
		t.Fatal("fixture bound", e)
	}
	if e = json.Unmarshal(b, out); e != nil {
		t.Fatal(e)
	}
}
func communityToyModel(t *testing.T, speech bool) *c1.ExperimentalDiarization {
	t.Helper()
	ctx := context.Background()
	var seg struct {
		Cases []struct {
			Config  c1.SegmentationLoadConfig
			Tensors map[string]struct {
				DType  string
				Shape  []int
				Values []float32
			}
		}
	}
	readCommunityFixture(t, "segmentation-load-reference.json.gz", &seg)
	c := seg.Cases[0]
	src := &communityFixtureSource{infos: map[string]safetensors.TensorInfo{}, values: map[string][]float32{}}
	names := make([]string, 0, len(c.Tensors))
	for n := range c.Tensors {
		names = append(names, n)
	}
	sort.Strings(names)
	offset := 0
	for _, n := range names {
		v := c.Tensors[n]
		src.infos[n] = safetensors.TensorInfo{DType: v.DType, Shape: v.Shape, DataOffsets: [2]int{offset, offset + 4*len(v.Values)}}
		offset += 4 * len(v.Values)
		src.values[n] = v.Values
	}
	for n, v := range src.values {
		if strings.Contains(n, "classifier") {
			clear(v)
			if strings.HasSuffix(n, "bias") {
				for i := range v {
					v[i] = -10
				}
				best := 0
				if speech {
					best = 1
				}
				v[best] = 10
			}
		}
	}
	checkpoint, e := c1.LoadSegmentationSource(ctx, src, c.Config)
	if e != nil {
		t.Fatal(e)
	}
	// Deliberate synthetic even impulse / zero odd filters satisfy the explicit
	// lowered-filter geometry. These are NOT trained filters or reference parity.
	filters := make([]float32, 80*251)
	for i := 0; i < 40; i++ {
		filters[i*251+125] = 1
	}
	segmentation, e := c1.NewExperimentalSegmentation(ctx, checkpoint, filters)
	if e != nil {
		t.Fatal(e)
	}
	var emb struct {
		Cases []struct {
			Config  c1.WeSpeakerResNetConfig
			Weights c1.WeSpeakerResNetWeights
		}
	}
	readCommunityFixture(t, "resnet-reference.json.gz", &emb)
	ec := emb.Cases[2]
	resnet, e := c1.NewWeSpeakerResNet34(ctx, ec.Config, ec.Weights)
	if e != nil {
		t.Fatal(e)
	}
	embedding, e := c1.NewExperimentalEmbedding(ctx, resnet)
	if e != nil {
		t.Fatal(e)
	}
	dim := ec.Config.EmbedDim
	mean := make([]float64, dim)
	matrix := make([]float64, dim*dim)
	phi := make([]float64, dim)
	for i := 0; i < dim; i++ {
		matrix[i*dim+i] = 1
		phi[i] = 1
	}
	plda, e := c1.NewPreparedPLDA(ctx, c1.PLDAConfig{InputDim: dim, ProjectedDim: dim, OutputDim: dim}, c1.PreparedPLDAWeights{Mean1: mean, Mean2: mean, LDA: matrix, Mu: mean, Transform: matrix, Phi: phi})
	if e != nil {
		t.Fatal(e)
	}
	model, e := c1.NewExperimentalDiarization(ctx, segmentation, embedding, plda)
	if e != nil {
		t.Fatal(e)
	}
	return model
}
func TestCommunityJobActualSyntheticPipeline(t *testing.T) {
	cfg := communityConfig()
	for _, speech := range []bool{false, true} {
		model := communityToyModel(t, speech)
		st, e := NewCommunity1Stage(model, cfg)
		if e != nil {
			t.Fatal(e)
		}
		s, _ := openTest(t)
		job := createTest(t, s)
		job, e = s.Run(context.Background(), job.ID, config, []Stage{fixturePCMStage(2960), textStage("asr-windows", "synthetic ASR"), st}, nil)
		if e != nil || job.Status != Complete {
			t.Fatal(speech, job, e)
		}
		r, e := s.OpenCheckpoint(context.Background(), job.ID, "diarization")
		raw := readAll(t, r, e)
		d, e := ReadDiarizationJSON(context.Background(), strings.NewReader(raw))
		if e != nil {
			t.Fatal(e)
		}
		want := "silence"
		if speech {
			want = "single-training-row"
		}
		if d.Path != want || len(d.AmbiguousFrames) != 0 || d.Policy.TiePolicy != c1.RejectAmbiguousTies {
			t.Fatal(d)
		}
		// Compare every retained field with a direct call through the same actual
		// synthetic model, without accepting a replacement timestamp convention.
		direct, e := model.RunPCM(context.Background(), communityZeroPCM{}, 2960, cfg.PCM, cfg.SegmentationModes, cfg.EmbeddingMode)
		if e != nil {
			t.Fatal(e)
		}
		expected, e := diarizationDocument(context.Background(), direct, cfg.PCM, 2960, d.StageKey)
		if e != nil || !reflect.DeepEqual(expected, d) {
			t.Fatal("adapter changed raw model output", e)
		}
		other := createTest(t, s)
		other, e = s.Run(context.Background(), other.ID, config, []Stage{fixturePCMStage(2960), textStage("asr-windows", "synthetic ASR"), st}, nil)
		if e != nil {
			t.Fatal(e)
		}
		r, e = s.OpenCheckpoint(context.Background(), other.ID, "diarization")
		if got := readAll(t, r, e); got != raw {
			t.Fatal("nondeterministic fresh job output")
		}
	}
}

func TestCommunitySyntheticClusteredAndCountOverride(t *testing.T) {
	cfg := communityConfig()
	model := communityToyModel(t, true)
	for _, samples := range []int{2960, 3361} {
		c := cfg
		if samples == 2960 {
			c.PCM.NumSpeakers = 5
		}
		st, e := NewCommunity1Stage(model, c)
		if e != nil {
			t.Fatal(e)
		}
		s, _ := openTest(t)
		job := createTest(t, s)
		job, e = s.Run(context.Background(), job.ID, config, []Stage{fixturePCMStage(samples), st}, nil)
		if e != nil {
			t.Fatal(samples, e)
		}
		r, e := s.OpenCheckpoint(context.Background(), job.ID, "diarization")
		d, e := ReadDiarizationJSON(context.Background(), strings.NewReader(readAll(t, r, e)))
		if e != nil {
			t.Fatal(e)
		}
		if samples == 2960 && (d.Path != "single-training-row" || d.ConstraintSatisfied || d.Policy.NumSpeakers != 5) {
			t.Fatal(d)
		}
		if samples == 3361 && (d.Path != "clustered" || !d.ConstraintSatisfied) {
			t.Fatal(d)
		}
	}
}
func TestCommunityCancelDrainsOwnedCall(t *testing.T) {
	s, _ := openTest(t)
	job := createTest(t, s)
	entered := make(chan struct{})
	release := make(chan struct{})
	st := community1Stage(communityConfig(), func(ctx context.Context, _ c1.DiarizationPCMReader, _ int64) (*c1.DiarizationPCMResult, error) {
		close(entered)
		<-ctx.Done()
		<-release
		return nil, ctx.Err()
	})
	done := make(chan error, 1)
	go func() {
		_, e := s.Run(context.Background(), job.ID, config, []Stage{fixturePCMStage(3361), st}, nil)
		done <- e
	}()
	<-entered
	if !s.Cancel(job.ID) {
		t.Fatal("cancel")
	}
	if e := s.Close(); !errors.Is(e, ErrBusy) {
		t.Fatal(e)
	}
	select {
	case e := <-done:
		t.Fatal("returned before drain", e)
	default:
	}
	close(release)
	if e := <-done; !errors.Is(e, context.Canceled) {
		t.Fatal(e)
	}
	job, e := s.Get(job.ID)
	if e != nil || job.Status != Cancelled {
		t.Fatal(job, e)
	}
}

type communityZeroPCM struct{}

func (communityZeroPCM) ReadSamplesAt(ctx context.Context, dst []float32, _ int64) (int, error) {
	if e := ctx.Err(); e != nil {
		return 0, e
	}
	clear(dst)
	return len(dst), nil
}

func TestCommunityActualModelCancellationAndRetry(t *testing.T) {
	model := communityToyModel(t, true)
	cfg := communityConfig()
	s, _ := openTest(t)
	job := createTest(t, s)
	cancelOnce := true
	var cancel context.CancelFunc
	infer := func(ctx context.Context, r c1.DiarizationPCMReader, total int64) (*c1.DiarizationPCMResult, error) {
		return model.RunPCMObserved(ctx, r, total, cfg.PCM, cfg.SegmentationModes, cfg.EmbeddingMode, func(stage string, _ int) error {
			if cancelOnce && stage == "segmentation" {
				cancelOnce = false
				cancel()
			}
			return nil
		})
	}
	st := community1Stage(cfg, infer)
	ctx, c := context.WithCancel(context.Background())
	cancel = c
	defer c()
	job, e := s.Run(ctx, job.ID, config, []Stage{fixturePCMStage(2960), textStage("asr-windows", "saved"), st}, nil)
	if !errors.Is(e, context.Canceled) || job.Status != Cancelled || len(job.Checkpoints) != 2 {
		t.Fatal(job, e)
	}
	job, e = s.Run(context.Background(), job.ID, config, []Stage{fixturePCMStage(2960), textStage("asr-windows", "saved"), st}, nil)
	if e != nil || job.Status != Complete {
		t.Fatal(job, e)
	}
}
