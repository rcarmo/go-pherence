//go:build linux

package pockettts

import (
	"context"
	"encoding/json"
	"errors"
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
	"reflect"
	"strings"
	"sync"
	"testing"
)

func TestTrainingCompactEmbeddingRewriteOwnership(t *testing.T) {
	source := embeddingRewriteFixture()
	bindings := map[string]train.ResidentTapeBinding{"table": {Parameter: 3, Gradient: 2}}
	before, _ := json.Marshal(source)
	a, b, c, err := compactResidentEmbeddingTape(source, bindings, 1024)
	if err != nil {
		t.Fatal(err)
	}
	other, _, _, err := compactResidentEmbeddingTape(source, bindings, 1024)
	if err != nil {
		t.Fatal(err)
	}
	if len(a.Tensors) != 6 || len(a.Commands) != 2 || b["table"].Parameter != 3 || b["table"].Gradient != 5 || c["table"].Index != 4 || a.Commands[0].Mode != 1 || a.Commands[1].Mode != 3 || a.Commands[1].Output != 5 || !reflect.DeepEqual(a.Outputs, []int{2, 4, 5}) {
		t.Fatal("source IDs/order/compact roots")
	}
	if a.Tensors[5].Shape[0] != 6 {
		t.Fatal("dense temporary extent")
	}
	out := interpretFlowTape(t, a)
	assertSliceClose(t, "public dense unchanged", out[0], interpretFlowTape(t, source)[0], 2e-6)
	compact := out[2]
	dense := make([]float32, 8)
	idx := a.Tensors[4].Values
	for slot := 0; slot < 3; slot++ {
		if idx[slot] < 0 {
			if !zeroTrainingSeed(compact[slot*2 : (slot+1)*2]) {
				t.Fatal("padding")
			}
			continue
		}
		copy(dense[int(idx[slot])*2:], compact[slot*2:(slot+1)*2])
	}
	assertSliceClose(t, "compact dense equivalence", dense, out[0], 2e-6)
	a.Commands[1].Inputs[0] = -1
	a.Tensors[4].Values[0] = -1
	b["table"] = train.ResidentTapeBinding{}
	after, _ := json.Marshal(source)
	if string(before) != string(after) || other.Commands[1].Inputs[0] != 0 || other.Tensors[4].Values[0] != 0 || bindings["table"].Parameter != 3 {
		t.Fatal("source/result ownership")
	}
	if err := train.ValidateResidentCompactTapeBindings(other, map[string]train.ResidentTapeBinding{"table": {Parameter: 3, Gradient: 5}}, c, 1024); err != nil {
		t.Fatal(err)
	}
}
func TestTrainingCompactEmbeddingRewriteAdmission(t *testing.T) {
	for _, kind := range []string{"source", "inventory", "budget", "combinedbudget", "graph"} {
		t.Run(kind, func(t *testing.T) {
			source := embeddingRewriteFixture()
			b := map[string]train.ResidentTapeBinding{"table": {Parameter: 3, Gradient: 2}}
			budget := int64(1024)
			switch kind {
			case "source":
				source.Tensors = append(source.Tensors, train.TensorSpec{Shape: []int{1}, Values: []float32{float32(math.NaN())}})
			case "inventory":
				b["bad"] = train.ResidentTapeBinding{Parameter: 3, Gradient: 2}
			case "budget":
				budget = 99
			case "combinedbudget":
				budget = 163 // full100+index40+compact24 =164
			case "graph":
				for len(source.Tensors) < 4095 {
					source.Tensors = append(source.Tensors, train.TensorSpec{Shape: []int{1}, Values: []float32{0}})
				}
				budget = 1 << 20
			}
			before, _ := json.Marshal(source)
			_, _, _, err := compactResidentEmbeddingTape(source, b, budget)
			if err == nil {
				t.Fatal("bad compact rewrite")
			}
			after, _ := json.Marshal(source)
			if string(before) != string(after) {
				t.Fatal("failed source mutated")
			}
		})
	}
	// No direct bound mode1 VJP: retain the admitted ordinary graph unchanged.
	source := train.Tape{Tensors: []train.TensorSpec{{Shape: []int{1}, Values: []float32{7}}, {Shape: []int{1}, Values: []float32{0}}}, Outputs: []int{1}}
	b := map[string]train.ResidentTapeBinding{"p": {Parameter: 0, Gradient: 1}}
	got, gb, c, err := compactResidentEmbeddingTape(source, b, 8)
	if err != nil || c != nil || !reflect.DeepEqual(got, source) || !reflect.DeepEqual(b, gb) {
		t.Fatal("no-op", err)
	}
}
func TestTrainingCompactEmbeddingRewriteConcurrent(t *testing.T) {
	source := embeddingRewriteFixture()
	b := map[string]train.ResidentTapeBinding{"table": {Parameter: 3, Gradient: 2}}
	var wg sync.WaitGroup
	for worker := 0; worker < 4; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 10; i++ {
				p, b, c, err := compactResidentEmbeddingTape(source, b, 1024)
				if err != nil {
					t.Error(err)
					return
				}
				p.Tensors[c["table"].Index].Values[0] = -1
				b["table"] = train.ResidentTapeBinding{}
				p.Commands[1].Inputs[0] = -1
			}
		}()
	}
	wg.Wait()
	if source.Commands[0].Mode != 1 || source.Commands[0].Inputs[0] != 0 || b["table"].Parameter != 3 {
		t.Fatal("shared source")
	}
}

// The wrapper records actual pruned tape dispatch, injects compact-boundary
// errors, and delegates every native-like lifecycle to the existing CPU mock.
type compactRecordingDriver struct {
	*hostResidentGraphDriver
	indexed, dense, compactValidations, compactCalls int
	fail                                             string
}

func (d *compactRecordingDriver) NewZeroBoundTape(ctx context.Context, p train.Tape, b map[int]train.ResidentBuffer, maxBytes int64) (train.ResidentBoundTape, error) {
	for _, c := range p.Commands {
		if c.Op == "embedding" {
			if c.Mode == 1 || c.Mode == 2 {
				d.dense++
			}
			if c.Mode == 3 {
				d.indexed++
				if err := train.ValidateCompactEmbeddingIndex(p.Tensors[c.Inputs[1]].Values, c.Rows, c.Parts); err != nil {
					d.t.Fatal(err)
				}
			}
		}
	}
	owner, err := d.hostResidentGraphDriver.NewZeroBoundTape(ctx, p, b, maxBytes)
	if owner == nil {
		return nil, err
	}
	return &compactRecordingTape{owner, d, p}, err
}

type compactRecordingTape struct {
	train.ResidentBoundTape
	driver *compactRecordingDriver
	source train.Tape
}

func (t *compactRecordingTape) OutputBuffer(id int) (train.ResidentBuffer, error) {
	if t.driver.fail == "index-output" {
		for _, c := range t.source.Commands {
			if c.Op == "embedding" && c.Mode == 3 && c.Inputs[1] == id {
				return nil, errors.New("compact index output")
			}
		}
	}
	return t.ResidentBoundTape.OutputBuffer(id)
}
func (d *compactRecordingDriver) ValidateCompactEmbeddingAccumulate(ctx context.Context, o, p, c, i train.ResidentBuffer, r, w, v int, s float32) error {
	d.compactValidations++
	if d.fail == "preflight" {
		return errors.New("compact preflight")
	}
	return d.hostResidentGraphDriver.ValidateCompactEmbeddingAccumulate(ctx, o, p, c, i, r, w, v, s)
}
func (d *compactRecordingDriver) CompactEmbeddingAccumulate(ctx context.Context, o, p, c, i train.ResidentBuffer, r, w, v int, s float32) error {
	d.compactCalls++
	if d.fail == "execution" {
		return errors.New("compact accumulation execution")
	}
	return d.hostResidentGraphDriver.CompactEmbeddingAccumulate(ctx, o, p, c, i, r, w, v, s)
}
func TestTrainingResidentEmbeddingCompactRouting(t *testing.T) {
	for _, scope := range []string{"ptx-resident-embeddingcompact-hybrid-v1", "vulkan-resident-embeddingcompact-hybrid-v1", "ptx-resident-embeddingindex-hybrid-v1", "ptx-resident-forwardresults-hybrid-v1"} {
		r, b, d := residentGraphRunner(t, scope)
		rec := &compactRecordingDriver{hostResidentGraphDriver: d}
		b.Driver = rec
		if _, err := r.Step(context.Background()); err != nil {
			t.Fatal(err)
		}
		if b.embeddingCompactEnabled() {
			if rec.indexed != 1 || rec.dense != 0 || rec.compactCalls != 1 || rec.compactValidations != 1 {
				t.Fatal("compact route", rec.indexed, rec.dense, rec.compactCalls)
			}
		} else if rec.indexed != 0 || rec.dense != 1 || rec.compactCalls != 0 {
			t.Fatal("legacy route")
		}
		if b.Close() != nil {
			t.Fatal("close")
		}
	}
}
func TestTrainingResidentEmbeddingCompactFailureBoundary(t *testing.T) {
	for _, kind := range []string{"preflight", "execution", "index-output"} {
		r, b, d := residentGraphRunner(t, "ptx-resident-embeddingcompact-hybrid-v1")
		rec := &compactRecordingDriver{hostResidentGraphDriver: d, fail: kind}
		b.Driver = rec
		before, raw := mustFullState(t, r.trainer), snapshotRunFixture(t, r.run)
		if _, err := r.Step(context.Background()); err == nil {
			t.Fatal("failure accepted", kind)
		}
		if !reflect.DeepEqual(before, mustFullState(t, r.trainer)) || string(raw) != string(snapshotRunFixture(t, r.run)) {
			t.Fatal("published partial update")
		}
		if kind == "preflight" && rec.compactCalls != 0 {
			t.Fatal("preflight mutated compact")
		}
		tapes := d.tapes
		if _, err := r.Step(context.Background()); err == nil || d.tapes != tapes || b.Close() == nil {
			t.Fatal("retry/owner release")
		}
	}
}

// Does not expose the compact interface, unlike the inner test driver.
type compactMissingDriver struct {
	train.ResidentDriver
	train.ResidentTapeDriver
	train.ResidentReuseTapeDriver
	train.ResidentZeroTapeDriver
}

func TestTrainingResidentEmbeddingCompactCapability(t *testing.T) {
	r, b, d := residentGraphRunner(t, "ptx-resident-embeddingcompact-hybrid-v1")
	b.Driver = &compactMissingDriver{d, d, d, d}
	err := b.PrepareTrainingStep(context.Background(), r.trainer)
	if err == nil || !strings.Contains(err.Error(), "compact embedding") || b.state != nil || d.allocations != 0 || d.tapes != 0 {
		t.Fatal("capability before device allocation", err)
	}
}

func TestTrainingResidentEmbeddingCompactPublicDenseOutput(t *testing.T) {
	r, b, d := residentGraphRunner(t, "ptx-resident-embeddingcompact-hybrid-v1")
	rec := &compactRecordingDriver{hostResidentGraphDriver: d}
	b.Driver = rec
	if err := b.PrepareTrainingStep(context.Background(), r.trainer); err != nil {
		t.Fatal(err)
	}
	m := r.trainer.FlowLM
	batch := FlowLMTrainingBatch{Frames: 1, NormalizedLatents: make([]float32, m.LatentDim), TextTokens: []uint32{1, 1}}
	seed := make([]float32, (2+len(batch.TextTokens))*m.Hidden)
	for i := range seed {
		seed[i] = float32(i%9-4) * .013
	}
	p, err := m.BuildConditionerTrainingTape(batch, seed, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	id := p.ResidentBindings["flow_lm.conditioner.embed.weight"].Gradient
	expected := interpretFlowTape(t, p.Tape)
	var want []float32
	for i, output := range p.Tape.Outputs {
		if output == id {
			want = expected[i]
		}
	}
	out, err := b.runResidentTape(p.Tape, p.ResidentBindings, []int{id}, true, 1)
	if err != nil {
		t.Fatal(err)
	}
	assertSliceClose(t, "explicit public dense gradient preserved", out[0], want, 2e-6)
	if rec.indexed != 1 || rec.dense != 1 || rec.compactCalls != 1 {
		t.Fatal("requested dense result pruned or accumulated twice")
	}
	// Isolated mock call has not published a trainer step; active Close stays
	// guarded. Clean state directly only in this CPU fixture.
	if b.Close() == nil {
		t.Fatal("active boundary")
	}
	if b.state.Close() != nil {
		t.Fatal("mock state cleanup")
	}
	b.active = false
	if b.Close() != nil {
		t.Fatal("close")
	}
}
func TestTrainingResidentEmbeddingCompactExpandedBudget(t *testing.T) {
	r, b, d := residentGraphRunner(t, "ptx-resident-embeddingcompact-hybrid-v1")
	if err := b.PrepareTrainingStep(context.Background(), r.trainer); err != nil {
		t.Fatal(err)
	}
	m := r.trainer.FlowLM
	batch := FlowLMTrainingBatch{Frames: 1, NormalizedLatents: make([]float32, m.LatentDim), TextTokens: []uint32{1, 1}}
	seed := make([]float32, (2+len(batch.TextTokens))*m.Hidden)
	p, err := m.BuildConditionerTrainingTape(batch, seed, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	b.MaxTapeBytes = tapeLogicalBytes(p.Tape)
	before := d.tapes
	if _, err := b.EvaluateConditioner(m, batch, seed, true); err == nil || d.tapes != before || b.terminal == nil {
		t.Fatal("expanded source budget bypass")
	}
	if b.Close() == nil {
		t.Fatal("failed owner")
	}
}

func TestTrainingResidentEmbeddingCompactStoragePlan(t *testing.T) {
	const rows, width, vocabulary = 16, 16, 4096
	ids := make([]float32, rows)
	for i := range ids {
		ids[i] = float32(i % 5)
	}
	source := train.Tape{Tensors: []train.TensorSpec{
		{Shape: []int{vocabulary * width}, Values: make([]float32, vocabulary*width)},
		{Shape: []int{rows * width}, Values: make([]float32, rows*width)},
		{Shape: []int{rows}, Values: ids},
		{Shape: []int{vocabulary * width}},
	}, Commands: []train.Command{{Op: "embedding", Inputs: []int{1, 2}, Output: 3, Rows: rows, Width: width, Parts: vocabulary, Mode: 1}}, Outputs: []int{3}}
	bindings := map[string]train.ResidentTapeBinding{"table": {Parameter: 0, Gradient: 3}}
	p, b, c, err := compactResidentEmbeddingTape(source, bindings, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	legacy, _, err := train.PruneTape(source, []int{3}, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	roots := []int{b["table"].Gradient, c["table"].Index}
	compact, _, err := train.PruneTape(p, roots, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	lp, err := train.PlanZeroTapeStorage(legacy, 1<<20, nil)
	if err != nil {
		t.Fatal(err)
	}
	cp, err := train.PlanZeroTapeStorage(compact, 1<<20, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, op := range compact.Commands {
		if op.Op == "embedding" && op.Mode != 3 {
			t.Fatal("dense VJP retained")
		}
	}
	if cp.Bytes >= lp.Bytes {
		t.Fatal("compact plan no reduction", lp.Bytes, cp.Bytes)
	}
	t.Logf("synthetic rows=%d width=%d vocabulary=%d: fullsource logical %d->%d, pruned logical %d->%d, planned storage %d->%d; no native peak/speed claim", rows, width, vocabulary, tapeLogicalBytes(source), tapeLogicalBytes(p), tapeLogicalBytes(legacy), tapeLogicalBytes(compact), lp.Bytes, cp.Bytes)
}

func TestTrainingResidentEmbeddingCompactCrossIndexRestore(t *testing.T) {
	r, b, _ := residentGraphRunner(t, "ptx-resident-embeddingcompact-hybrid-v1")
	if _, err := r.Step(context.Background()); err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	limits := TrainingBundleLimits{1 << 20, 1 << 16}
	if _, err := r.Checkpoint(dir, "compact", limits); err != nil {
		t.Fatal(err)
	}
	for _, scope := range []string{"ptx-resident-embeddingindex-hybrid-v1", "ptx-resident-forwardresults-hybrid-v1", "vulkan-resident-embeddingcompact-hybrid-v1"} {
		old, ob, _ := residentGraphRunner(t, scope)
		before, raw := mustFullState(t, old.trainer), snapshotRunFixture(t, old.run)
		if old.Restore(dir, limits) == nil {
			t.Fatal("cross-scope restore", scope)
		}
		if !reflect.DeepEqual(before, mustFullState(t, old.trainer)) || string(raw) != string(snapshotRunFixture(t, old.run)) {
			t.Fatal("restore mutated")
		}
		if ob.Close() != nil {
			t.Fatal("close")
		}
	}
	if b.Close() != nil {
		t.Fatal("close")
	}
}
