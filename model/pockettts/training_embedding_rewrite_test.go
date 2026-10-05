package pockettts

import (
	"context"
	"encoding/json"
	"math"
	"reflect"
	"strings"
	"sync"
	"testing"

	train "github.com/rcarmo/go-pherence/backends/training"
)

func embeddingRewriteFixture() train.Tape {
	return train.Tape{Tensors: []train.TensorSpec{{Shape: []int{6}, Values: []float32{1, 2, 3, 4, 5, 6}}, {Shape: []int{3}, Values: []float32{2, 0, 2}}, {Shape: []int{8}}, {Shape: []int{8}, Values: make([]float32, 8)}}, Commands: []train.Command{{Op: "embedding", Inputs: []int{0, 1}, Output: 2, Rows: 3, Width: 2, Parts: 4, Mode: 1}}, Outputs: []int{2}}
}

func TestTrainingEmbeddingRewriteOwnership(t *testing.T) {
	source := embeddingRewriteFixture()
	before, _ := json.Marshal(source)
	bindings := map[string]train.ResidentTapeBinding{"table": {Parameter: 3, Gradient: 2}}
	a, err := indexResidentEmbeddingTape(source, bindings, 1024)
	if err != nil {
		t.Fatal(err)
	}
	b, err := indexResidentEmbeddingTape(source, bindings, 1024)
	if err != nil {
		t.Fatal(err)
	}
	if len(a.Tensors) != 5 || a.Commands[0].Output != 2 || a.Commands[0].Inputs[0] != 0 || a.Commands[0].Inputs[1] != 4 || a.Commands[0].Mode != 2 || !reflect.DeepEqual(a.Outputs, source.Outputs) {
		t.Fatal("IDs/order/roots")
	}
	want := []float32{0, 1, 1, 3, 3, 1, 0, 2}
	if !reflect.DeepEqual(a.Tensors[4].Values, want) {
		t.Fatal("index", a.Tensors[4].Values)
	}
	assertSliceClose(t, "rewritten VJP", interpretFlowTape(t, a)[0], interpretFlowTape(t, source)[0], 2e-6)
	a.Commands[0].Inputs[0] = -1
	a.Tensors[4].Values[0] = -1
	after, _ := json.Marshal(source)
	if string(before) != string(after) || b.Commands[0].Inputs[0] != 0 || b.Tensors[4].Values[0] != 0 {
		t.Fatal("source/crosscall alias")
	}
	// No embedding VJP leaves the admitted read-only graph unchanged.
	constant := train.Tape{Tensors: []train.TensorSpec{{Shape: []int{1}, Values: []float32{7}}, {Shape: []int{1}, Values: []float32{0}}}, Outputs: []int{1}}
	got, err := indexResidentEmbeddingTape(constant, map[string]train.ResidentTapeBinding{"p": {Parameter: 0, Gradient: 1}}, 8)
	if err != nil || !reflect.DeepEqual(constant, got) {
		t.Fatal("no-op", err)
	}
	// Mode0 still consumes ordinary IDs; already-indexed commands are not rebuilt.
	g := source
	g.Commands = append([]train.Command(nil), source.Commands...)
	g.Commands[0].Mode = 0
	g.Tensors = append([]train.TensorSpec(nil), source.Tensors...)
	g.Tensors[0] = train.TensorSpec{Shape: []int{8}, Values: make([]float32, 8)}
	g.Tensors[2].Shape = []int{6}
	g.Tensors[3] = train.TensorSpec{Shape: []int{6}, Values: make([]float32, 6)}
	got, err = indexResidentEmbeddingTape(g, bindings, 1024)
	if err != nil || !reflect.DeepEqual(g, got) {
		t.Fatal("gather changed", err)
	}
	got, err = indexResidentEmbeddingTape(b, bindings, 1024)
	if err != nil || !reflect.DeepEqual(b, got) {
		t.Fatal("indexed twice", err)
	}
}

func TestTrainingEmbeddingRewriteAdmission(t *testing.T) {
	for _, kind := range []string{"source", "index-budget", "fullsource-budget", "inventory", "graph-ceiling"} {
		t.Run(kind, func(t *testing.T) {
			source := embeddingRewriteFixture()
			budget := int64(1024)
			bindings := map[string]train.ResidentTapeBinding{"table": {Parameter: 3, Gradient: 2}}
			switch kind {
			case "source":
				source.Tensors = append(source.Tensors, train.TensorSpec{Shape: []int{1}, Values: []float32{float32(math.NaN())}})
			case "index-budget":
				budget = 100 // source100B passes; rewritten132B fails at131.
				budget = 131
			case "fullsource-budget":
				budget = 99
			case "inventory":
				bindings["bad"] = train.ResidentTapeBinding{Parameter: 0, Gradient: 1}
			case "graph-ceiling":
				for len(source.Tensors) < 4096 {
					source.Tensors = append(source.Tensors, train.TensorSpec{Shape: []int{1}, Values: []float32{0}})
				}
				budget = 1 << 20
			}
			before, _ := json.Marshal(source)
			_, err := indexResidentEmbeddingTape(source, bindings, budget)
			if err == nil {
				t.Fatal("invalid rewrite", kind)
			}
			if kind == "index-budget" && !strings.Contains(err.Error(), "byte ceiling") {
				t.Fatal(err)
			}
			after, _ := json.Marshal(source)
			if string(before) != string(after) {
				t.Fatal("rejected source mutated")
			}
		})
	}
}

func TestTrainingEmbeddingRewriteConcurrent(t *testing.T) {
	source := embeddingRewriteFixture()
	var wg sync.WaitGroup
	for w := 0; w < 4; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 10; i++ {
				got, err := indexResidentEmbeddingTape(source, map[string]train.ResidentTapeBinding{"table": {Parameter: 3, Gradient: 2}}, 1024)
				if err != nil {
					t.Error(err)
					return
				}
				got.Commands[0].Inputs[0] = -1
				got.Tensors[4].Values[0] = -1
			}
		}()
	}
	wg.Wait()
	if source.Commands[0].Inputs[0] != 0 || source.Tensors[1].Values[0] != 2 {
		t.Fatal("source mutated")
	}
}

// Record the actual pruned resident graph reaching the mock device, not merely
// the pre-pruning rewrite. Original source admission happens before this hook.
type embeddingRecordingDriver struct {
	*hostResidentGraphDriver
	indexed, legacy int
}

func (d *embeddingRecordingDriver) NewZeroBoundTape(ctx context.Context, tape train.Tape, parameters map[int]train.ResidentBuffer, maxBytes int64) (train.ResidentBoundTape, error) {
	for _, c := range tape.Commands {
		if c.Op == "embedding" {
			if c.Mode == 1 {
				d.legacy++
			}
			if c.Mode == 2 {
				d.indexed++
				if err := train.ValidateEmbeddingIndex(tape.Tensors[c.Inputs[1]].Values, c.Rows, c.Parts); err != nil {
					d.t.Fatal(err)
				}
			}
		}
	}
	return d.hostResidentGraphDriver.NewZeroBoundTape(ctx, tape, parameters, maxBytes)
}

func TestTrainingResidentEmbeddingIndexRouting(t *testing.T) {
	for _, scope := range []string{"ptx-resident-embeddingindex-hybrid-v1", "vulkan-resident-embeddingindex-hybrid-v1", "ptx-resident-forwardresults-hybrid-v1"} {
		r, b, d := residentGraphRunner(t, scope)
		record := &embeddingRecordingDriver{hostResidentGraphDriver: d}
		b.Driver = record
		if _, err := r.Step(context.Background()); err != nil {
			t.Fatal(err)
		}
		if b.embeddingIndexEnabled() {
			if record.indexed != 1 || record.legacy != 0 {
				t.Fatal("indexed dispatch", record.indexed, record.legacy)
			}
		} else if record.indexed != 0 || record.legacy != 1 {
			t.Fatal("legacy dispatch")
		}
		if b.Close() != nil {
			t.Fatal("close")
		}
	}
}

func TestTrainingResidentEmbeddingIndexBudgetBeforeDispatch(t *testing.T) {
	r, b, d := residentGraphRunner(t, "ptx-resident-embeddingindex-hybrid-v1")
	if err := b.PrepareTrainingStep(context.Background(), r.trainer); err != nil {
		t.Fatal(err)
	}
	source := embeddingRewriteFixture()
	b.MaxTapeBytes = 131 // fullsource100 passes; index raises combined to132.
	before, accum := d.tapes, d.accumulations
	_, err := b.runResidentTape(source, map[string]train.ResidentTapeBinding{"table": {Parameter: 3, Gradient: 2}}, source.Outputs, true, 1)
	if err == nil || d.tapes != before || d.accumulations != accum {
		t.Fatal("late index admission mutated driver", err)
	}
	b.fail(err)
	if b.Close() == nil {
		t.Fatal("failed owner released")
	}
}
