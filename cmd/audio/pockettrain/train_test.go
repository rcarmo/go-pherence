//go:build linux

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/rcarmo/go-pherence/model/pockettts"
)

// Fake only the update/publish boundary, never the numerical training code.
// The existing tiny-model runner gates exercise the concrete implementation.
type boundaryRunner struct {
	step       uint64
	calls      int
	saved      []uint64
	events     []string
	afterStep  func(int)
	beforeStep func(int)
	stepErr    error
	saveErr    error
}

func (r *boundaryRunner) Progress() (uint64, uint64, int) { return r.step, 0, int(r.step) }
func (r *boundaryRunner) Step(ctx context.Context) (pockettts.TrainingStepMetrics, error) {
	r.calls++
	if r.beforeStep != nil {
		r.beforeStep(r.calls)
	}
	if err := ctx.Err(); err != nil {
		return pockettts.TrainingStepMetrics{}, err
	}
	if r.stepErr != nil {
		return pockettts.TrainingStepMetrics{}, r.stepErr
	}
	r.step++
	r.events = append(r.events, "update")
	if r.afterStep != nil {
		r.afterStep(r.calls)
	}
	return pockettts.TrainingStepMetrics{Loss: 1}, nil
}
func (r *boundaryRunner) Checkpoint(_ string, _ string, _ pockettts.TrainingBundleLimits) (pockettts.TrainingCheckpointBinding, error) {
	r.saved = append(r.saved, r.step)
	r.events = append(r.events, "save")
	return pockettts.TrainingCheckpointBinding{Step: r.step}, r.saveErr
}

type boundaryWriter struct {
	runner *boundaryRunner
	err    error
	data   bytes.Buffer
}

func (w *boundaryWriter) Write(p []byte) (int, error) {
	w.runner.events = append(w.runner.events, "report")
	if w.err != nil {
		return 0, w.err
	}
	return w.data.Write(p)
}
func boundaryGeneration(step uint64) (string, error) { return "fixed-generation", nil }
func runBoundary(ctx context.Context, r *boundaryRunner, steps, every int, w io.Writer) error {
	return runPreparedUpdates(ctx, r, "unused", steps, every, pockettts.TrainingBundleLimits{MaxModelBytes: 1 << 20, MaxSidecarBytes: 1 << 16}, w, boundaryGeneration)
}
func TestPreparedLoopCheckpointBeforeReport(t *testing.T) {
	r := &boundaryRunner{step: 5}
	w := &boundaryWriter{runner: r}
	if err := runBoundary(context.Background(), r, 3, 2, w); err != nil {
		t.Fatal(err)
	}
	if len(r.saved) != 2 || r.saved[0] != 6 || r.saved[1] != 8 {
		t.Fatal("absolute interval/final checkpoints", r.saved)
	}
	want := []string{"update", "save", "report", "update", "report", "update", "save", "report"}
	if strings.Join(r.events, ",") != strings.Join(want, ",") {
		t.Fatal(r.events)
	}
	dec := json.NewDecoder(&w.data)
	for _, wantSaved := range []uint64{6, 6, 8} {
		var line struct {
			Step       uint64 `json:"step"`
			Checkpoint uint64 `json:"checkpoint_step"`
		}
		if err := dec.Decode(&line); err != nil || line.Checkpoint != wantSaved {
			t.Fatal(line, err)
		}
	}
}
func TestPreparedLoopCancellationSavesDirty(t *testing.T) {
	for _, after := range []int{0, 1, 2} {
		t.Run(string(rune('0'+after)), func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			r := &boundaryRunner{afterStep: func(call int) {
				if call == after {
					cancel()
				}
			}}
			if after == 0 {
				cancel()
			}
			if err := runBoundary(ctx, r, 2, 10, io.Discard); !errors.Is(err, context.Canceled) {
				t.Fatal(err)
			}
			if after == 0 {
				if r.calls != 0 || len(r.saved) != 0 {
					t.Fatal("cancel before update mutated state")
				}
				return
			}
			if len(r.saved) != 1 || r.saved[0] != uint64(after) {
				t.Fatal("dirty boundary not saved once", r.saved)
			}
		})
	}
}
func TestPreparedLoopCancellationRacingStep(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	r := &boundaryRunner{beforeStep: func(call int) {
		if call == 2 {
			cancel()
		}
	}}
	if err := runBoundary(ctx, r, 3, 10, io.Discard); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if r.calls != 2 || r.step != 1 || len(r.saved) != 1 || r.saved[0] != 1 {
		t.Fatal("racing cancellation lost prior boundary", r.saved, r.calls)
	}
}
func TestPreparedLoopInvalidAdmission(t *testing.T) {
	r := &boundaryRunner{}
	for _, tc := range []struct {
		ctx          context.Context
		r            preparedUpdateRunner
		out          io.Writer
		steps, every int
		gen          func(uint64) (string, error)
	}{
		{nil, r, io.Discard, 1, 1, boundaryGeneration},
		{context.Background(), nil, io.Discard, 1, 1, boundaryGeneration},
		{context.Background(), r, nil, 1, 1, boundaryGeneration},
		{context.Background(), r, io.Discard, 0, 1, boundaryGeneration},
		{context.Background(), r, io.Discard, 1, 0, boundaryGeneration},
		{context.Background(), r, io.Discard, 1, 1, nil},
	} {
		if err := runPreparedUpdates(tc.ctx, tc.r, "unused", tc.steps, tc.every, pockettts.TrainingBundleLimits{}, tc.out, tc.gen); err == nil {
			t.Fatal("invalid loop admitted")
		}
	}
	if r.calls != 0 || len(r.saved) != 0 {
		t.Fatal("invalid loop side effects")
	}
}
func TestPreparedLoopOutputFailureSavesDirty(t *testing.T) {
	outputErr := errors.New("broken output")
	for _, every := range []int{1, 10} {
		r := &boundaryRunner{}
		w := &boundaryWriter{runner: r, err: outputErr}
		if err := runBoundary(context.Background(), r, 2, every, w); !errors.Is(err, outputErr) {
			t.Fatal(err)
		}
		if r.calls != 1 || len(r.saved) != 1 || r.saved[0] != 1 {
			t.Fatal("output failure lost update or retried checkpoint", r.saved, r.calls)
		}
	}
}
func TestPreparedLoopPublicationFailureStops(t *testing.T) {
	publishErr := &pockettts.TrainingBundlePublishedError{Err: errors.New("directory sync")}
	r := &boundaryRunner{saveErr: publishErr}
	w := &boundaryWriter{runner: r}
	if err := runBoundary(context.Background(), r, 2, 1, w); err != publishErr {
		t.Fatal("typed publication error lost", err)
	}
	if r.calls != 1 || len(r.saved) != 1 || w.data.Len() != 0 {
		t.Fatal("publication retried/reported", r.events)
	}
	ctx, cancel := context.WithCancel(context.Background())
	r = &boundaryRunner{saveErr: publishErr, afterStep: func(int) { cancel() }}
	if err := runBoundary(ctx, r, 2, 10, io.Discard); !errors.Is(err, context.Canceled) || !errors.Is(err, publishErr) {
		t.Fatal("cancellation/publication causes lost", err)
	}
	if len(r.saved) != 1 {
		t.Fatal("uncertain publication retried")
	}
}
func TestPreparedLoopOutputAndSaveErrors(t *testing.T) {
	outputErr, saveErr := errors.New("broken output"), errors.New("checkpoint failure")
	r := &boundaryRunner{saveErr: saveErr}
	w := &boundaryWriter{runner: r, err: outputErr}
	if err := runBoundary(context.Background(), r, 2, 10, w); !errors.Is(err, outputErr) || !errors.Is(err, saveErr) {
		t.Fatal("causes lost", err)
	}
	if len(r.saved) != 1 || r.calls != 1 {
		t.Fatal("checkpoint retried", r.events)
	}
}
func TestPreparedLoopUpdateFailureNeverSaved(t *testing.T) {
	updateErr := errors.New("possibly mutated Adam")
	r := &boundaryRunner{stepErr: updateErr}
	if err := runBoundary(context.Background(), r, 2, 1, io.Discard); err != updateErr {
		t.Fatal(err)
	}
	if len(r.saved) != 0 || r.calls != 1 {
		t.Fatal("failed update checkpointed/retried")
	}
}
func TestPreparedLoopGenerationFailureStops(t *testing.T) {
	r := &boundaryRunner{}
	entropyErr := errors.New("entropy failure")
	err := runPreparedUpdates(context.Background(), r, "unused", 1, 1, pockettts.TrainingBundleLimits{}, io.Discard, func(uint64) (string, error) { return "", entropyErr })
	if err != entropyErr || len(r.saved) != 0 || r.calls != 1 {
		t.Fatal(err, r.events)
	}
}

func TestPreparedRowPolicyAdmission(t *testing.T) {
	fixture := func() pockettts.PreparedTrainingRow {
		return pockettts.PreparedTrainingRow{Batch: pockettts.FlowLMTrainingBatch{Frames: 3, VoiceFrames: 1, NormalizedLatents: []float32{1, 2, 3, 4, 0, float32(math.Copysign(0, -1))}, VoiceLatents: []float32{1, 2}, TextTokens: []uint32{0, 3}}, ValidFrames: 2}
	}
	if err := validatePreparedRow(fixture(), 2, 4); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name   string
		mutate func(*pockettts.PreparedTrainingRow)
	}{
		{"no-padding", func(r *pockettts.PreparedTrainingRow) { r.ValidFrames = 3 }},
		{"two-padding", func(r *pockettts.PreparedTrainingRow) { r.ValidFrames = 1 }},
		{"nonzero-padding", func(r *pockettts.PreparedTrainingRow) { r.Batch.NormalizedLatents[5] = 1 }},
		{"missing-target", func(r *pockettts.PreparedTrainingRow) { r.Batch.NormalizedLatents = nil }},
		{"voice-shape", func(r *pockettts.PreparedTrainingRow) { r.Batch.VoiceLatents = nil }},
		{"nonfinite-target", func(r *pockettts.PreparedTrainingRow) { r.Batch.NormalizedLatents[0] = float32(math.NaN()) }},
		{"nonfinite-voice", func(r *pockettts.PreparedTrainingRow) { r.Batch.VoiceLatents[0] = float32(math.Inf(1)) }},
		{"token-range", func(r *pockettts.PreparedTrainingRow) { r.Batch.TextTokens[0] = 4 }},
		{"token-max", func(r *pockettts.PreparedTrainingRow) { r.Batch.TextTokens[0] = math.MaxUint32 }},
		{"empty-text", func(r *pockettts.PreparedTrainingRow) { r.Batch.TextTokens = nil }},
		{"token-ceiling", func(r *pockettts.PreparedTrainingRow) { r.Batch.TextTokens = make([]uint32, 513) }},
		{"no-voice", func(r *pockettts.PreparedTrainingRow) { r.Batch.VoiceFrames = 0 }},
		{"voice-ceiling", func(r *pockettts.PreparedTrainingRow) { r.Batch.VoiceFrames = 63 }},
		{"frame-ceiling", func(r *pockettts.PreparedTrainingRow) { r.Batch.Frames = 376 }},
		{"one-frame", func(r *pockettts.PreparedTrainingRow) { r.Batch.Frames = 1 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := fixture()
			tc.mutate(&r)
			if err := validatePreparedRow(r, 2, 4); err == nil {
				t.Fatal("invalid row admitted")
			}
		})
	}
	for _, dims := range [][2]int{{0, 4}, {2, 0}, {math.MaxInt, 4}} {
		if err := validatePreparedRow(fixture(), dims[0], dims[1]); err == nil {
			t.Fatal("invalid/overflow dimensions admitted")
		}
	}
}

func TestPreparedTrainingCLIAdmission(t *testing.T) {
	for _, args := range [][]string{{"-backend=ptx"}, {"-backend=vulkan"}, {"-steps=0"}, {"-allow-experimental-sampler", "-steps=0"}} {
		if err := trainPrepared(context.Background(), args, io.Discard); err == nil {
			t.Fatal("unsafe options admitted")
		}
	}
	if !trainingRequested([]string{"-mode", "train"}) || !trainingRequested([]string{"--mode=train"}) || trainingRequested([]string{"-mode=preflight"}) {
		t.Fatal("mode dispatch")
	}
}
func TestPreparedExportExclusivePublication(t *testing.T) {
	dir := t.TempDir()
	temp := filepath.Join(dir, "temp")
	dest := filepath.Join(dir, "export")
	if err := os.WriteFile(temp, []byte("new"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(dest, []byte("old"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := publishNewExport(temp, dest); err == nil {
		t.Fatal("raced output replaced")
	}
	data, err := os.ReadFile(dest)
	if err != nil || string(data) != "old" {
		t.Fatal("old output changed", err)
	}
	if err := os.Remove(dest); err != nil {
		t.Fatal(err)
	}
	if err := publishNewExport(temp, dest); err != nil {
		t.Fatal(err)
	}
	data, err = os.ReadFile(dest)
	if err != nil || string(data) != "new" {
		t.Fatal("new output absent", err)
	}
}
func TestPreparedCorpusJSON(t *testing.T) {
	for _, s := range []string{`{}`, `{"version":2,"rows":[]}`, `{"version":1,"rows":[],"unknown":1}`, `{"version":1,"rows":[]} {}`} {
		if _, err := parsePrepared([]byte(s)); err == nil {
			t.Fatal("invalid corpus admitted")
		}
	}
	rows, err := parsePrepared([]byte(`{"version":1,"rows":[{"batch":{"Frames":3,"VoiceFrames":0,"NormalizedLatents":[0,1,2,3,4,5],"VoiceLatents":[],"TextTokens":[1]},"valid_frames":2}]}`))
	if err != nil || len(rows) != 1 || rows[0].ValidFrames != 2 {
		t.Fatal(rows, err)
	}
	if err := trainPrepared(context.Background(), []string{"-backend=ptx", "-config=/missing"}, io.Discard); err == nil || !strings.Contains(err.Error(), "not implemented") {
		t.Fatal("backend rejection did not precede file access")
	}
}
