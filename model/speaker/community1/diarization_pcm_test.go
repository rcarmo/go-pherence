package community1

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"reflect"
	"sync"
	"testing"
)

type diarizationTestReader struct {
	samples    []float32
	starts     []int64
	lengths    []int
	fail       error
	short      bool
	wrappedEOF bool
}

func (r *diarizationTestReader) ReadSamplesAt(ctx context.Context, dst []float32, start int64) (int, error) {
	if e := ctx.Err(); e != nil {
		return 0, e
	}
	r.starts = append(r.starts, start)
	r.lengths = append(r.lengths, len(dst))
	if r.fail != nil {
		return 0, r.fail
	}
	if start >= int64(len(r.samples)) {
		return 0, io.EOF
	}
	n := copy(dst, r.samples[start:])
	if r.short && n > 0 {
		n--
	}
	if n < len(dst) {
		return n, io.EOF
	}
	if r.wrappedEOF {
		return n, fmt.Errorf("final read: %w", io.EOF)
	}
	return n, nil
}
func TestDiarizationWindowPlan(t *testing.T) {
	for _, c := range []struct {
		n          int64
		count      int
		last       int64
		valid, pad int
	}{{1, 1, 0, 1, 159999}, {159999, 1, 0, 159999, 1}, {160000, 1, 0, 160000, 0}, {160001, 2, 16000, 144001, 15999}, {175999, 2, 16000, 159999, 1}, {176000, 2, 16000, 160000, 0}, {176001, 3, 32000, 144001, 15999}, {480000, 21, 320000, 160000, 0}} {
		w, e := PlanDiarizationWindows(c.n, 160000, 16000)
		if e != nil || len(w) != c.count {
			t.Fatal(c, e)
		}
		last := w[len(w)-1]
		if last != (DiarizationWindow{c.last, c.valid, c.pad}) {
			t.Fatal(c, last)
		}
		for i, x := range w {
			if x.Start != int64(i)*16000 || x.Samples+x.Padding != 160000 {
				t.Fatal("schedule")
			}
		}
	}
	for _, c := range [][3]int64{{0, 160000, 16000}, {-1, 160000, 16000}, {14400*16000 + 1, 160000, 16000}, {1, 0, 1}, {1, 160001, 1}, {1, 100, 101}, {8193, 1, 1}, {1000, 100, 0}} {
		if w, e := PlanDiarizationWindows(c[0], int(c[1]), int(c[2])); w != nil || e == nil {
			t.Fatal("accepted", c)
		}
	}
	// A long recording can pass the old 4096-window ceiling without relaxing
	// the 8192-window admission bound.
	if w, e := PlanDiarizationWindows(160000+4869*16000, 160000, 16000); e != nil || len(w) != 4870 {
		t.Fatal("4870", e)
	}
	// Count bound is inclusive and rejected before allocation.
	if w, e := PlanDiarizationWindows(160000+8191*16000, 160000, 16000); e != nil || len(w) != 8192 {
		t.Fatal("8192", e)
	}
	if _, e := PlanDiarizationWindows(160001+8191*16000, 160000, 16000); e == nil {
		t.Fatal("8193")
	}
}

// This is only a model-free geometry preflight for the 81m TEF recording.
// It does not qualify speaker accuracy, run neural inference, or predict time.
func TestTEFLongDiarizationGeometryPreflight(t *testing.T) {
	const samples int64 = 78059864 // retained decoded PCM extent before cancellation
	windows, err := PlanDiarizationWindows(samples, 160000, 16000)
	if err != nil || len(windows) != 4870 {
		t.Fatal("window plan", len(windows), err)
	}
	grid, err := sincNetGrid(160000, 10) // pinned Community-1 SincNet stride
	if err != nil || grid.Frames != 589 || grid.Step != 270 {
		t.Fatal("segmentation grid", grid, err)
	}
	const local, dim = 3, 256
	if int64(len(windows))*int64(grid.Frames)*local > 1<<24 || len(windows)*local*dim > 1<<24 {
		t.Fatal("segmentation/embedding element bound")
	}
	cfg := ReconstructionConfig{
		Chunks: len(windows), Frames: grid.Frames, Speakers: local,
		Start: 0, ChunkDuration: 10, ChunkStep: 1,
		FrameDuration: float64(grid.ReceptiveField) / 16000,
		FrameStep:     float64(grid.Step) / 16000,
		MaxSpeakers:   64, TiePolicy: LowestIndexTies,
	}
	starts, frames, err := reconstructionGrid(cfg)
	if err != nil || frames < 280000 || frames > 300000 || len(starts) != len(windows) {
		t.Fatal("reconstruction grid", frames, err)
	}
	if int64(frames)*local > 1<<24 || int64(frames)*local*local > 1<<28 || frames*local > 1<<24 {
		t.Fatal("reconstruction/turn element bound")
	}
	if _, err := BinaryActivityToTurns(context.Background(), make([]uint8, frames*local), BinaryTurnConfig{
		Frames: frames, Speakers: local, FrameDuration: cfg.FrameDuration, FrameStep: cfg.FrameStep,
	}); err != nil {
		t.Fatal("turn geometry", err)
	}
}

func diarizationFixture(t *testing.T, speech bool) (*ExperimentalDiarization, DiarizationPCMConfig, SegmentationModes) {
	t.Helper()
	seg, _ := experimentalFixture(t)
	emb, _ := embeddingPCMFixture(t)
	// Deliberately controlled synthetic head for silence/single/multirow paths.
	clear(seg.checkpoint.head.classifier.Weight)
	for i := range seg.checkpoint.head.classifier.Bias {
		seg.checkpoint.head.classifier.Bias[i] = -10
	}
	best := 0
	if speech {
		best = 1
	}
	seg.checkpoint.head.classifier.Bias[best] = 10
	dim := emb.model.cfg.EmbedDim
	mean := make([]float64, dim)
	matrix := make([]float64, dim*dim)
	phi := make([]float64, dim)
	for i := 0; i < dim; i++ {
		matrix[i*dim+i] = 1
		phi[i] = 1
	}
	p, e := NewPreparedPLDA(context.Background(), PLDAConfig{dim, dim, dim}, PreparedPLDAWeights{mean, mean, matrix, mean, matrix, phi})
	if e != nil {
		t.Fatal(e)
	}
	m, e := NewExperimentalDiarization(context.Background(), seg, emb, p)
	if e != nil {
		t.Fatal(e)
	}
	return m, DiarizationPCMConfig{WindowSamples: 2960, StepSamples: 1600, MinimumEmbeddingSamples: 400, ExcludeOverlap: true, MinSpeakers: 1, MaxSpeakers: 4, AHCThreshold: .6, Fa: .07, Fb: .8, TiePolicy: LowestIndexTies}, SegmentationModes{SincNetSIMDFMA, LSTMSIMD, HeadSIMD}
}
func TestDiarizationOverlappedBranches(t *testing.T) {
	ctx := context.Background()
	seg := &SegmentationPCMResult{Classes: 1}
	trunk := &EmbeddingPCMFrames{samples: 1}
	started := make(chan string, 2)
	release := make(chan struct{})
	type branchResult struct {
		branches diarizationBranches
		err      error
	}
	finished := make(chan branchResult, 1)
	go func() {
		branches, err := runDiarizationBranches(ctx,
			func(context.Context) (*SegmentationPCMResult, error) {
				started <- "segmentation"
				<-release
				return seg, nil
			},
			func(context.Context) (*EmbeddingPCMFrames, error) {
				started <- "embedding"
				<-release
				return trunk, nil
			})
		finished <- branchResult{branches, err}
	}()
	first, second := <-started, <-started
	close(release)
	joined := <-finished
	if joined.err != nil || joined.branches.segmentation != seg || joined.branches.trunk != trunk || first == second {
		t.Fatal("branches did not overlap/join", first, second, joined)
	}
	cause := errors.New("branch failed")
	cancelled := make(chan error, 1)
	branches, err := runDiarizationBranches(ctx,
		func(context.Context) (*SegmentationPCMResult, error) { return nil, cause },
		func(branchCtx context.Context) (*EmbeddingPCMFrames, error) {
			<-branchCtx.Done()
			cancelled <- branchCtx.Err()
			return nil, branchCtx.Err()
		})
	if branches != (diarizationBranches{}) || !errors.Is(err, cause) || !errors.Is(err, context.Canceled) || !errors.Is(<-cancelled, context.Canceled) {
		t.Fatal("branch cancellation/error lost", branches, err)
	}
	for _, test := range []struct {
		ctx       context.Context
		segNil    bool
		embedNil  bool
		wantError bool
	}{{nil, false, false, true}, {ctx, true, false, true}, {ctx, false, true, true}} {
		branches, err = runDiarizationBranches(test.ctx,
			func(context.Context) (*SegmentationPCMResult, error) {
				if test.segNil {
					return nil, nil
				}
				return seg, nil
			},
			func(context.Context) (*EmbeddingPCMFrames, error) {
				if test.embedNil {
					return nil, nil
				}
				return trunk, nil
			})
		if (err != nil) != test.wantError || err != nil && branches != (diarizationBranches{}) {
			t.Fatal("invalid branch input/result", test, branches, err)
		}
	}
}

func TestExperimentalDiarizationComposition(t *testing.T) {
	for _, speech := range []bool{false, true} {
		for _, samples := range []int{2960, 3361} {
			m, cfg, modes := diarizationFixture(t, speech)
			pcm := make([]float32, samples)
			for i := range pcm {
				pcm[i] = float32(math.Sin(float64(i)*.071)) * .13
			}
			before := append([]float32(nil), pcm...)
			reader := &diarizationTestReader{samples: pcm}
			var stages []string
			got, e := m.RunPCMObserved(context.Background(), reader, int64(samples), cfg, modes, WeSpeakerBlockSIMD, func(s string, _ int) error { stages = append(stages, s); return nil })
			if e != nil {
				t.Fatal(speech, samples, e)
			}
			wantPath := "silence"
			if speech {
				wantPath = "single-training-row"
				if samples > 2960 {
					wantPath = "clustered"
				}
			}
			if got.Postprocess.Path != wantPath || len(reader.starts) != len(got.Windows) || !reflect.DeepEqual(before, pcm) {
				t.Fatal("path/read/ownership", got.Postprocess.Path)
			}
			var segments, embeddings []float32
			for _, w := range got.Windows {
				p := make([]float32, cfg.WindowSamples)
				copy(p, pcm[w.Start:w.Start+int64(w.Samples)])
				scores, e := m.segmentation.ForwardPCM(context.Background(), p, modes)
				if e != nil {
					t.Fatal(e)
				}
				ps, _ := NewPowerset(3, 2)
				binary, e := ps.Decode(context.Background(), scores.LogProbabilities, got.Grid.Frames, PowersetHard)
				if e != nil {
					t.Fatal(e)
				}
				segments = append(segments, binary...)
				mask, e := SelectEmbeddingMasks(context.Background(), binary, EmbeddingMaskConfig{got.Grid.Frames, 3, cfg.WindowSamples, 400, true})
				if e != nil {
					t.Fatal(e)
				}
				f, e := m.embedding.ForwardPCMFrames(context.Background(), p, WeSpeakerBlockSIMD)
				if e != nil {
					t.Fatal(e)
				}
				emb, e := m.embedding.EmbedFrames(context.Background(), f, mask.Masks, 3, got.Grid.Frames, WeSpeakerBlockSIMD)
				if e != nil {
					t.Fatal(e)
				}
				embeddings = append(embeddings, emb.Embeddings...)
			}
			if !reflect.DeepEqual(got.Segmentations, segments) || !reflect.DeepEqual(got.Embeddings, embeddings) {
				t.Fatal("manual neural/mask composition")
			}
			overlappedReader := &diarizationTestReader{samples: pcm}
			overlapped, e := m.RunPCMOverlapped(context.Background(), overlappedReader, int64(samples), cfg, modes, WeSpeakerBlockSIMD)
			if e != nil || !reflect.DeepEqual(overlapped, got) || !reflect.DeepEqual(overlappedReader.starts, reader.starts) || !reflect.DeepEqual(before, pcm) {
				t.Fatal("overlapped parity/read/ownership", e)
			}
			var observed []string
			var observedMu sync.Mutex
			overlapped, e = m.RunPCMOverlappedObserved(context.Background(), &diarizationTestReader{samples: pcm}, int64(samples), cfg, modes, WeSpeakerBlockSIMD, func(stage string, _ int) error {
				observedMu.Lock()
				observed = append(observed, stage)
				observedMu.Unlock()
				return nil
			})
			if e != nil || !reflect.DeepEqual(overlapped, got) || !reflect.DeepEqual(observed, stages) {
				t.Fatal("overlapped observed parity/order", observed, stages, e)
			}
			post := PostprocessConfig{Reconstruction: ReconstructionConfig{len(got.Windows), got.Grid.Frames, 3, 0, float64(cfg.WindowSamples) / 16000, float64(cfg.StepSamples) / 16000, float64(got.Grid.ReceptiveField) / 16000, float64(got.Grid.Step) / 16000, 4, LowestIndexTies}, EmbeddingDimension: got.EmbeddingDimension, MinSpeakers: 1, AHCThreshold: .6, Fa: .07, Fb: .8}
			want, e := PostprocessCommunity1(context.Background(), segments, embeddings, m.plda, post)
			if e != nil || !reflect.DeepEqual(got.Postprocess, want) {
				t.Fatal("postprocess composition", e)
			}
			if len(stages) < 4*len(got.Windows)+1 {
				t.Fatal("missing stages")
			}
			for i := range got.Windows {
				if !reflect.DeepEqual(stages[i*4:i*4+4], []string{"read", "segmentation", "masks", "embedding"}) {
					t.Fatal("stage order")
				}
			}
		}
	}
}
func TestExperimentalDiarizationRejectsAndCancels(t *testing.T) {
	m, cfg, modes := diarizationFixture(t, true)
	pcm := make([]float32, 3361)
	ctx := context.Background()
	sentinel := errors.New("reader/observer")
	for _, kind := range []string{"short", "read", "nil", "step", "window", "minimum", "speakers", "nan", "tie", "mode", "embeddingmode"} {
		reader := &diarizationTestReader{samples: pcm}
		var source DiarizationPCMReader = reader
		c, sm, em := cfg, modes, WeSpeakerBlockSIMD
		switch kind {
		case "short":
			reader.short = true
		case "read":
			reader.fail = sentinel
		case "nil":
			source = nil
		case "step":
			c.StepSamples = 0
		case "window":
			c.WindowSamples = 100
		case "minimum":
			c.MinimumEmbeddingSamples = 0
		case "speakers":
			c.MinSpeakers = 0
		case "nan":
			c.Fa = math.NaN()
		case "tie":
			c.TiePolicy = 99
		case "mode":
			sm.SincNet = SincNetScalar
		case "embeddingmode":
			em = 99
		}
		r, e := m.RunPCM(ctx, source, int64(len(pcm)), c, sm, em)
		if r != nil || e == nil {
			t.Fatal("accepted", kind)
		}
		if kind == "read" && !errors.Is(e, sentinel) {
			t.Fatal("reader error lost")
		}
		if kind != "read" && kind != "short" && len(reader.starts) != 0 {
			t.Fatal("invalid config read PCM", kind)
		}
	}
	for _, stop := range []string{"read", "segmentation", "masks", "embedding", "count", "filter", "ahc", "plda", "vbx", "centroids", "assignment", "reconstruction", "turns"} {
		for _, cancelled := range []bool{false, true} {
			cc, cancel := context.WithCancel(ctx)
			called := false
			r, e := m.RunPCMObserved(cc, &diarizationTestReader{samples: pcm}, int64(len(pcm)), cfg, modes, WeSpeakerBlockSIMD, func(stage string, _ int) error {
				if called {
					t.Error("continued after abort")
				}
				if stage == stop {
					called = true
					if cancelled {
						cancel()
						return nil
					}
					return sentinel
				}
				return nil
			})
			cancel()
			expected := sentinel
			if cancelled {
				expected = context.Canceled
			}
			if r != nil || !errors.Is(e, expected) || !called {
				t.Fatal(stop, cancelled, e)
			}
		}
	}
	var nilModel *ExperimentalDiarization
	cc, cancel := context.WithCancel(ctx)
	cancel()
	if r, e := nilModel.RunPCM(cc, nil, 0, cfg, modes, WeSpeakerBlockSIMD); r != nil || !errors.Is(e, context.Canceled) {
		t.Fatal("precancel")
	}
	if r, e := NewExperimentalDiarization(ctx, nil, m.embedding, nil); r != nil || e == nil {
		t.Fatal("bad constructor")
	}
	// Recovery after all cancellations.
	if _, e := m.RunPCM(ctx, &diarizationTestReader{samples: pcm}, int64(len(pcm)), cfg, modes, WeSpeakerBlockSIMD); e != nil {
		t.Fatal(e)
	}
}

func TestExperimentalDiarizationPolicyBoundaries(t *testing.T) {
	m, cfg, modes := diarizationFixture(t, true)
	ctx := context.Background()
	pcm := make([]float32, cfg.WindowSamples)
	// Underlying explicit NumSpeakers overrides min/max, including early single-row path.
	cfg.NumSpeakers = 5
	r, e := m.RunPCM(ctx, &diarizationTestReader{samples: pcm, wrappedEOF: true}, int64(len(pcm)), cfg, modes, WeSpeakerBlockSIMD)
	if e != nil || r.Postprocess.Path != "single-training-row" || r.Postprocess.ConstraintSatisfied {
		t.Fatal("override", e)
	}
	// Always-overlap speech has no clean training rows; finite empty-mask bias is
	// not a reason to invent training data. Mask selection and filter stay distinct.
	cfg.NumSpeakers = 0
	for i := range m.segmentation.checkpoint.head.classifier.Bias {
		m.segmentation.checkpoint.head.classifier.Bias[i] = -10
	}
	m.segmentation.checkpoint.head.classifier.Bias[4] = 10
	if r, e := m.RunPCM(ctx, &diarizationTestReader{samples: pcm}, int64(len(pcm)), cfg, modes, WeSpeakerBlockSIMD); r != nil || !errors.Is(e, ErrNoTrainingEmbeddings) {
		t.Fatal("overlap-only admission", e)
	}
}
