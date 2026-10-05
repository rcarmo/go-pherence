package pockettts

import (
	"reflect"
	"sync"
	"testing"

	train "github.com/rcarmo/go-pherence/backends/training"
)

// Only remapped commands acquire new ownership here. Unremapped source
// shapes/constants and the leading component commands keep their read-only
// borrowing contract; this test does not require those arrays to be cloned.
func checkCompositionOperandRegions(t *testing.T, got []train.Command, source []train.Command, mapping []int) {
	t.Helper()
	if len(got) != len(source) {
		t.Fatal("command count")
	}
	for i, c := range got {
		want := source[i]
		want.Inputs = append([]int(nil), want.Inputs...)
		for j, id := range want.Inputs {
			want.Inputs[j] = mapping[id]
		}
		want.Output = mapping[want.Output]
		if !reflect.DeepEqual(c, want) {
			t.Fatal("remap/order/mode/scales", i)
		}
		if cap(c.Inputs) != len(c.Inputs) {
			t.Fatal("operand capacity", i)
		}
	}
	// Hold deep copies so writes/appends cannot silently alter later commands.
	originals := make([][]int, len(got))
	sources := make([][]int, len(source))
	for i := range got {
		originals[i] = append([]int(nil), got[i].Inputs...)
		sources[i] = append([]int(nil), source[i].Inputs...)
	}
	for i, c := range got {
		if len(c.Inputs) == 0 {
			continue
		}
		c.Inputs[0] = -1
		grown := append(c.Inputs, -2)
		grown[0] = -3
		if c.Inputs[0] != -1 {
			t.Fatal("append reaches command", i)
		}
		for j := range got {
			if j != i && !reflect.DeepEqual(got[j].Inputs, originals[j]) {
				t.Fatal("write reaches sibling", i, j)
			}
			if !reflect.DeepEqual(source[j].Inputs, sources[j]) {
				t.Fatal("write reaches source", i, j)
			}
		}
		c.Inputs[0] = originals[i][0]
	}
}

func TestTrainingCompositionOperandDistill(t *testing.T) {
	ep, pp := distillCompositionSources(t)
	a, err := composeDistillTrainingTapes(ep, pp, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	b, err := composeDistillTrainingTapes(ep, pp, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	mapping := make([]int, len(pp.Tape.Tensors))
	for i := range mapping {
		mapping[i] = len(ep.Tape.Tensors) + i
	}
	for name, p := range pp.ResidentBindings {
		mapping[p.Parameter] = ep.ResidentBindings[name].Parameter
	}
	for i, id := range []int{ep.Tape.Outputs[1], ep.Tape.Outputs[2], ep.Tape.Outputs[0], ep.Tape.Outputs[4]} {
		mapping[pp.seedIDs[i]] = id
	}
	start := len(ep.Tape.Commands)
	checkCompositionOperandRegions(t, a.Tape.Commands[start:], pp.Tape.Commands, mapping)
	compareCompositionRetained(t, a.Tape, b.Tape, start)
}

func TestTrainingCompositionOperandForward(t *testing.T) {
	interpolation, p, e := forwardCompositionSources(t)
	a, err := composeDistillForwardTapes(interpolation, p, e, .5, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	b, err := composeDistillForwardTapes(interpolation, p, e, .5, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	pm := make([]int, len(p.Tape.Tensors))
	for i := range pm {
		pm[i] = len(interpolation.Tensors) + i
	}
	pm[p.inputValue] = interpolation.Outputs[0]
	start := len(interpolation.Commands)
	checkCompositionOperandRegions(t, a.Tape.Commands[start:start+len(p.Tape.Commands)], p.Tape.Commands, pm)
	em := make([]int, len(e.Tape.Tensors))
	for i := range em {
		em[i] = len(interpolation.Tensors) + len(p.Tape.Tensors) + 2 + i
	}
	em[e.inputValue] = a.Tape.Outputs[3]
	for name, ep := range e.ResidentBindings {
		em[ep.Parameter] = pm[p.ResidentBindings[name].Parameter]
	}
	endpointStart := start + len(p.Tape.Commands) + 2
	checkCompositionOperandRegions(t, a.Tape.Commands[endpointStart:], e.Tape.Commands, em)
	// Independent remap invocations cannot reach earlier append regions.
	saved := a.Tape.Commands[start].Inputs[0]
	a.Tape.Commands[endpointStart].Inputs[0] = -1
	if a.Tape.Commands[start].Inputs[0] != saved {
		t.Fatal("endpoint reaches primary")
	}
	a.Tape.Commands[endpointStart].Inputs[0] = b.Tape.Commands[endpointStart].Inputs[0]
	compareCompositionRetained(t, a.Tape, b.Tape, start)
}

func compareCompositionRetained(t *testing.T, a, b train.Tape, start int) {
	t.Helper()
	// The exact IDs/modes/scales are structural contract, not generated-signal
	// equality. Numerically interpreted results use the existing tolerance.
	av, bv := interpretFlowTape(t, a), interpretFlowTape(t, b)
	for i, v := range av {
		assertSliceClose(t, "composition values", bv[i], v, 2e-6)
	}
	saved := b.Commands[start].Inputs[0]
	b.Commands[start].Inputs[0] = -1
	again := interpretFlowTape(t, a)
	b.Commands[start].Inputs[0] = saved
	for i, v := range av {
		assertSliceClose(t, "retained composition", again[i], v, 2e-6)
	}
}

func TestTrainingCompositionOperandConcurrent(t *testing.T) {
	ep, pp := distillCompositionSources(t)
	interpolation, p, e := forwardCompositionSources(t)
	var wg sync.WaitGroup
	for worker := 0; worker < 4; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 5; i++ {
				d, err := composeDistillTrainingTapes(ep, pp, 64<<20)
				if err != nil {
					t.Error(err)
					return
				}
				f, err := composeDistillForwardTapes(interpolation, p, e, .5, 64<<20)
				if err != nil {
					t.Error(err)
					return
				}
				d.Tape.Commands[len(ep.Tape.Commands)].Inputs[0] = -1
				f.Tape.Commands[len(interpolation.Commands)].Inputs[0] = -1
			}
		}()
	}
	wg.Wait()
	// Admission after all mutated independent results also checks that shared
	// immutable sources remain valid after concurrent composition and writes.
	if _, err := composeDistillTrainingTapes(ep, pp, 64<<20); err != nil {
		t.Fatal(err)
	}
	if _, err := composeDistillForwardTapes(interpolation, p, e, .5, 64<<20); err != nil {
		t.Fatal(err)
	}
}
