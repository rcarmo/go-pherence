package pockettts

import (
	"context"
	"fmt"
	"reflect"
	"testing"

	train "github.com/rcarmo/go-pherence/backends/training"
)

func adapterOracleInventory(tape train.Tape, bindings map[string]train.ResidentTapeBinding, expected map[string][]float32) error {
	if len(bindings) != len(expected) {
		return fmt.Errorf("oracle gradient name cardinality")
	}
	counts := map[int]int{}
	for _, id := range tape.Outputs {
		counts[id]++
	}
	gradIDs, paramIDs := map[int]bool{}, map[int]bool{}
	for name, b := range bindings {
		want, ok := expected[name]
		if !ok {
			return fmt.Errorf("foreign oracle gradient %s", name)
		}
		if b.Parameter < 0 || b.Parameter >= len(tape.Tensors) || b.Gradient < 0 || b.Gradient >= len(tape.Tensors) || b.Parameter == b.Gradient || gradIDs[b.Gradient] || paramIDs[b.Parameter] || counts[b.Gradient] != 1 {
			return fmt.Errorf("duplicate/undeclared oracle gradient root %s", name)
		}
		gradIDs[b.Gradient] = true
		paramIDs[b.Parameter] = true
		n, e := tape.Tensors[b.Gradient].Elements()
		if e != nil || n != len(want) || !finiteF32(want) {
			return fmt.Errorf("oracle gradient extent/finite %s", name)
		}
	}
	for name := range expected {
		if _, ok := bindings[name]; !ok {
			return fmt.Errorf("missing oracle gradient %s", name)
		}
	}
	return nil
}
func adapterComputedInventory(computed, expected map[string][]float32) error {
	if len(computed) != len(expected) {
		return fmt.Errorf("computed oracle name cardinality")
	}
	for name, want := range expected {
		got, ok := computed[name]
		if !ok || len(got) != len(want) || !finiteF32(got) || !finiteF32(want) {
			return fmt.Errorf("computed oracle missing/extent/finite %s", name)
		}
	}
	return nil
}
func flowAdapterExpected(raw map[string][]float32) map[string][]float32 {
	out := map[string][]float32{}
	for name, v := range raw {
		out["flow_lm.flow_net."+name] = v
	}
	return out
}
func transformerAdapterExpected(raw map[string][]float32) map[string][]float32 {
	out := map[string][]float32{}
	for name, v := range raw {
		if len(name) >= 6 && name[:6] == "final." {
			name = "out_norm." + name[6:]
		}
		out["flow_lm."+name] = v
	}
	return out
}
func flowLMAdapterExpected(raw map[string][]float32) (map[string][]float32, error) {
	if _, ok := raw["out_eos.weight"]; !ok {
		return nil, fmt.Errorf("fixture missing named EOS weight")
	}
	if _, ok := raw["out_eos.bias"]; !ok {
		return nil, fmt.Errorf("fixture missing named EOS bias")
	}
	out := map[string][]float32{}
	for name, v := range raw {
		if name == "out_eos.weight" || name == "out_eos.bias" {
			continue
		}
		out["flow_lm."+name] = v
	}
	return out, nil
}
func adapterStepRootRoles(p StepBackboneTrainingTape, batch FlowLMTrainingBatch, in StepReductionInputs, s WeightSampleInputs, roots []int) error {
	count := 5
	if batch.VoiceFrames > 0 {
		count = 6
	}
	if len(roots) != count {
		return fmt.Errorf("step root role count")
	}
	want := append([]int(nil), p.Tape.Outputs[2:6]...)
	if batch.VoiceFrames > 0 {
		want = append([]int(nil), p.Tape.Outputs[2:7]...)
	}
	tailID := -1
	for _, c := range p.Tape.Commands {
		if c.Op == "vector" && c.Mode == 1 && c.Offset == (batch.Frames-1)*in.Latent && len(c.Inputs) == 2 && c.Inputs[0] == p.Tape.Outputs[1] {
			n, _ := p.Tape.Tensors[c.Output].Elements()
			if n == in.Latent {
				if tailID >= 0 {
					return fmt.Errorf("ambiguous tail role")
				}
				tailID = c.Output
			}
		}
	}
	if tailID < 0 {
		return fmt.Errorf("missing tail role")
	}
	want = append(want, tailID)
	if !reflect.DeepEqual(roots, want) {
		return fmt.Errorf("step ordered root roles")
	}
	writers := map[int]train.Command{}
	declared := map[int]bool{}
	for _, c := range p.Tape.Commands {
		writers[c.Output] = c
	}
	for _, id := range p.Tape.Outputs {
		declared[id] = true
	}
	seen := map[int]bool{}
	sizes := []int{in.Frames, in.Frames, 5}
	if batch.VoiceFrames > 0 {
		sizes = append(sizes, len(batch.VoiceLatents))
	}
	sizes = append(sizes, len(batch.NormalizedLatents), in.Latent)
	for i, id := range roots {
		if id < 0 || id >= len(p.Tape.Tensors) || !declared[id] || seen[id] {
			return fmt.Errorf("step role undeclared/duplicate")
		}
		seen[id] = true
		n, e := p.Tape.Tensors[id].Elements()
		if e != nil || n != sizes[i] {
			return fmt.Errorf("step role extent %d", i)
		}
	}
	// Logvar roots must scale their own admitted raw seed, not a same-shaped
	// neighbouring root. Verify the actual vector's source values and mode.
	for i, seed := range [][]float32{s.DiagonalSeed, s.DistillSeed} {
		c := writers[roots[i]]
		if c.Op != "vector" || c.Mode != 0 || len(c.Inputs) != 2 || !reflect.DeepEqual(p.Tape.Tensors[c.Inputs[1]].Values, seed) {
			return fmt.Errorf("step logvar seed role %d", i)
		}
	}
	tail := writers[roots[len(roots)-1]]
	if tail.Op != "vector" || tail.Mode != 1 || tail.Offset != (batch.Frames-1)*in.Latent || len(tail.Inputs) != 2 || tail.Inputs[0] != p.Tape.Outputs[1] {
		return fmt.Errorf("step direct-tail source role")
	}
	normalized := writers[roots[len(roots)-2]]
	if normalized.Op != "vector" || normalized.Mode != 0 {
		return fmt.Errorf("step normalized result role")
	}
	metrics := writers[roots[2]]
	if metrics.Op != "vector" || metrics.Mode != 0 {
		return fmt.Errorf("step metrics role")
	}
	return nil
}
func TestPocketAdapterCorrectionEffectiveBindings(t *testing.T) {
	for _, kind := range []string{"gather-range", "gather-fraction", "dense-index", "compact-index", "eos-label", "eos-active", "lsd-logvar", "lsd-cutoff", "time-frequency"} {
		t.Run(kind, func(t *testing.T) {
			var tape train.Tape
			var id int
			var invalid []float32
			switch kind {
			case "gather-range", "gather-fraction":
				tape = train.Tape{Tensors: []train.TensorSpec{{Shape: []int{2}, Values: []float32{3, 5}}, {Shape: []int{1}, Values: []float32{0}}, {Shape: []int{1}}}, Commands: []train.Command{{Op: "embedding", Inputs: []int{0, 1}, Output: 2, Mode: 0, Rows: 1, Width: 1, Parts: 2}}, Outputs: []int{2}}
				id = 1
				invalid = []float32{2}
				if kind == "gather-fraction" {
					invalid[0] = .5
				}
			case "dense-index":
				ix, e := train.BuildEmbeddingIndex([]float32{0}, 2)
				if e != nil {
					t.Fatal(e)
				}
				tape = train.Tape{Tensors: []train.TensorSpec{{Shape: []int{1}, Values: []float32{2}}, {Shape: []int{len(ix)}, Values: ix}, {Shape: []int{2}}}, Commands: []train.Command{{Op: "embedding", Inputs: []int{0, 1}, Output: 2, Mode: 2, Rows: 1, Width: 1, Parts: 2}}, Outputs: []int{2}}
				id = 1
				invalid = append([]float32(nil), ix...)
				invalid[0] = 2
			case "compact-index":
				ix, e := train.BuildCompactEmbeddingIndex([]float32{0}, 2)
				if e != nil {
					t.Fatal(e)
				}
				tape = train.Tape{Tensors: []train.TensorSpec{{Shape: []int{1}, Values: []float32{2}}, {Shape: []int{len(ix)}, Values: ix}, {Shape: []int{1}}}, Commands: []train.Command{{Op: "embedding", Inputs: []int{0, 1}, Output: 2, Mode: 3, Rows: 1, Width: 1, Parts: 2}}, Outputs: []int{2}}
				id = 1
				invalid = append([]float32(nil), ix...)
				invalid[1] = 1
			case "eos-label", "eos-active":
				tape = train.Tape{Tensors: []train.TensorSpec{{Shape: []int{1}, Values: []float32{.2}}, {Shape: []int{1}, Values: []float32{1}}, {Shape: []int{1}}}, Commands: []train.Command{{Op: "eos", Inputs: []int{0, 1}, Output: 2, Mode: 0, Rows: 1}}, Outputs: []int{2}}
				id = 1
				invalid = []float32{.5}
				if kind == "eos-active" {
					invalid[0] = -1
				}
			case "time-frequency":
				tape = train.Tape{Tensors: []train.TensorSpec{{Shape: []int{1}, Values: []float32{1}}, {Shape: []int{2}, Values: []float32{1, 1}}, {Shape: []int{2}, Values: []float32{0, 0}}, {Shape: []int{2}, Values: []float32{0, 0}}, {Shape: []int{2}}}, Commands: []train.Command{{Op: "time-embedding", Inputs: []int{0, 1, 2, 3}, Output: 4, Mode: 0}}, Outputs: []int{4}}
				id = 0
				invalid = []float32{9000}
			case "lsd-logvar", "lsd-cutoff":
				mode := uint32(0)
				outN := 3
				v := float32(0)
				if kind == "lsd-cutoff" {
					mode = 2
					outN = 1
					v, _ = train.LSDPredictionCutoff(1, 0)
				}
				tape = train.Tape{Tensors: []train.TensorSpec{{Shape: []int{1}, Values: []float32{.1}}, {Shape: []int{1}, Values: []float32{0}}, {Shape: []int{1}, Values: []float32{v}}, {Shape: []int{outN}}}, Commands: []train.Command{{Op: "lsd-loss", Inputs: []int{0, 1, 2}, Output: 3, Mode: mode, Scale: 0}}, Outputs: []int{3}}
				id = 2
				invalid = []float32{81}
				if kind == "lsd-cutoff" {
					invalid[0] = v * .5
				}
			}
			if e := tape.Validate(1 << 20); e != nil {
				t.Fatal("invalid original fixture", kind, e)
			}
			before := append([]float32(nil), invalid...)
			original := pocketAdapterFreeze(tape)
			callbacks := 0
			out, ledger, e := runPocketAdapter(context.Background(), tape, map[int][]float32{id: invalid}, nil, pocketAdapterOptions{reuse: true, before: func(int) { callbacks++ }})
			if e == nil || out != nil || callbacks != 0 || ledger.SlotPayload != 0 || ledger.Dispatched != 0 {
				t.Fatal("semantic binding admitted", kind, e, ledger)
			}
			if !reflect.DeepEqual(invalid, before) || !reflect.DeepEqual(tape, original) {
				t.Fatal("admission mutated caller")
			}
		})
	}
	// Ordinary affine-like P/data constants remain rebindable; not all replacement
	// sources are semantic metadata. Execute the copied values, never original P.
	tape := train.Tape{Tensors: []train.TensorSpec{{Shape: []int{2}, Values: []float32{1, 2}}, {Shape: []int{2}, Values: []float32{3, 4}}, {Shape: []int{2}}}, Commands: []train.Command{{Op: "vector", Inputs: []int{0, 1}, Output: 2, Scale: 1}}, Outputs: []int{2}}
	out, _, e := runPocketAdapter(context.Background(), tape, map[int][]float32{0: {7, 8}}, nil, pocketAdapterOptions{reuse: true})
	if e != nil {
		t.Fatal(e)
	}
	var stats pocketAdapterErrors
	stats.compare(t, "valid ordinary P replacement", out[0], []float32{10, 12}, 2e-6)
	gather := train.Tape{Tensors: []train.TensorSpec{{Shape: []int{2}, Values: []float32{3, 5}}, {Shape: []int{1}, Values: []float32{0}}, {Shape: []int{1}}}, Commands: []train.Command{{Op: "embedding", Inputs: []int{0, 1}, Output: 2, Mode: 0, Rows: 1, Width: 1, Parts: 2}}, Outputs: []int{2}}
	out, _, e = runPocketAdapter(context.Background(), gather, map[int][]float32{1: {1}}, nil, pocketAdapterOptions{reuse: true})
	if e != nil {
		t.Fatal("valid semantic replacement", e)
	}
	stats.compare(t, "valid bound gather ID1", out[0], []float32{5}, 2e-6)
	gather.Tensors[1].Values[0] = 2
	if out, ledger, e := runPocketAdapter(context.Background(), gather, map[int][]float32{1: {0}}, nil, pocketAdapterOptions{reuse: true}); e == nil || out != nil || ledger.SlotPayload != 0 {
		t.Fatal("invalid original repaired by rebound", e)
	}
}
func TestPocketAdapterCorrectionOracleInventory(t *testing.T) {
	tape := train.Tape{Tensors: []train.TensorSpec{{Shape: []int{1}, Values: []float32{0}}, {Shape: []int{1}, Values: []float32{2}}, {Shape: []int{1}, Values: []float32{0}}, {Shape: []int{1}, Values: []float32{3}}}, Outputs: []int{1, 3}}
	expected := map[string][]float32{"a": {2}, "b": {3}}
	good := map[string]train.ResidentTapeBinding{"a": {Parameter: 0, Gradient: 1}, "b": {Parameter: 2, Gradient: 3}}
	for _, kind := range []string{"missing", "foreign", "duplicate-root", "undeclared", "duplicate-output", "extent"} {
		t.Run(kind, func(t *testing.T) {
			b := map[string]train.ResidentTapeBinding{}
			for n, v := range good {
				b[n] = v
			}
			x := pocketAdapterFreeze(tape)
			switch kind {
			case "missing":
				delete(b, "a")
			case "foreign":
				b["z"] = b["a"]
				delete(b, "a")
			case "duplicate-root":
				v := b["b"]
				v.Gradient = 1
				b["b"] = v
			case "undeclared":
				x.Outputs = x.Outputs[:1]
			case "duplicate-output":
				x.Outputs = append(x.Outputs, 1)
			case "extent":
				x.Tensors[1].Shape = []int{2}
			}
			if adapterOracleInventory(x, b, expected) == nil {
				t.Fatal("invalid oracle root/name set accepted", kind)
			}
		})
	}
	if e := adapterOracleInventory(tape, good, expected); e != nil {
		t.Fatal(e)
	}
	if adapterComputedInventory(map[string][]float32{"a": {2}}, expected) == nil {
		t.Fatal("missing computed result accepted")
	}
}
