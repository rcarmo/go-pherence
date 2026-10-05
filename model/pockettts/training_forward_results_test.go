package pockettts

import (
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
	"testing"
)

func TestTrainingForwardResultsReference(t *testing.T) {
	for _, frames := range []int{1, 2, 7} {
		for _, voice := range []int{0, 1} {
			_, m := backboneOracle(t)
			batch := eosForwardBatch(frames, voice, []uint32{1, 1})
			mask := make([]bool, frames)
			for i := range mask {
				mask[i] = i < frames-1 || frames == 1
			}
			p, err := m.buildEOSForwardTrainingTape(batch, mask, .1, 64<<20)
			if err != nil {
				t.Fatal(err)
			}
			full := interpretFlowTape(t, p.Tape)
			q, err := buildForwardResultsTape(p, m, batch, 64<<20)
			if err != nil {
				t.Fatal(err)
			}
			all := interpretFlowTape(t, q.Tape)
			prefix := 1 + voice + len(batch.TextTokens)
			assertSliceClose(t, "audio hidden", all[len(all)-2], full[1][prefix*m.Hidden:], 2e-5)
			assertSliceClose(t, "mean", all[len(all)-3], full[len(full)-1], 2e-6)
			if all[len(all)-1][0] != 0 {
				t.Fatal("finite witness")
			}
			if len(q.ResidentBindings) != len(p.ResidentBindings) {
				t.Fatal("roots")
			}
			for name, b := range p.ResidentBindings {
				if q.ResidentBindings[name] != b {
					t.Fatal("changed bindings")
				}
			}
			bytes := tapeLogicalBytes(q.Tape)
			if _, err := buildForwardResultsTape(p, m, batch, bytes-1); err == nil {
				t.Fatal("tail/witness budget")
			}
			if _, err := buildForwardResultsTape(p, m, batch, bytes); err != nil {
				t.Fatal(err)
			}
		}
	}
}
func TestTrainingForwardResultsAdmission(t *testing.T) {
	_, m := backboneOracle(t)
	batch := eosForwardBatch(2, 1, []uint32{1, 1})
	p, err := m.buildEOSForwardTrainingTape(batch, []bool{true, false}, .1, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"source", "topology", "extent"} {
		q := p
		b := batch
		switch kind {
		case "source":
			q.Tape.Tensors = append(append([]train.TensorSpec(nil), p.Tape.Tensors...), train.TensorSpec{Shape: []int{1}, Values: []float32{float32(math.NaN())}})
		case "topology":
			b.Frames = 0
		case "extent":
			b.Frames++
		}
		if _, err := buildForwardResultsTape(q, m, b, 64<<20); err == nil {
			t.Fatal(kind)
		}
	}
}

func TestTrainingForwardResultsFiniteWitness(t *testing.T) {
	// Independent admitted sources generate invalid discarded outputs. The
	// witness must retain them under pruning without overflowing finite values.
	for _, x := range []float32{0, math.MaxFloat32, -math.MaxFloat32, float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN())} {
		g := &transformerTapeGraph{constants: map[int][]float32{}, zeros: map[int]bool{}, params: map[string]int{}}
		id := g.tensor([]int{1, 2})
		if isFinite(x) {
			g.constants[id] = []float32{x, x}
		} else {
			a := g.constant([]int{1, 1}, []float32{math.MaxFloat32})
			xt := g.zero(1, 1)
			w := g.constant([]int{2, 1}, []float32{2, 2})
			if math.IsInf(float64(x), -1) {
				g.constants[w] = []float32{-2, -2}
			}
			if math.IsNaN(float64(x)) {
				tmp := g.command("linear-forward", []int{a, xt, g.constant([]int{1, 1}, []float32{2}), g.zero(1)}, []int{1, 1}, 0)
				a = tmp
				w = g.zero(2, 1)
			}
			g.tape.Commands = append(g.tape.Commands, train.Command{Op: "linear-forward", Inputs: []int{a, xt, w, g.zero(2)}, Output: id})
		}
		witness := appendTrainingFiniteWitness(g, []int{id})
		g.tape.Outputs = []int{witness}
		built, err := materializeSequenceTape(g, nil, 64<<20)
		if err != nil {
			t.Fatal(err)
		}
		v := interpretFlowTape(t, built.Tape)
		got := v[0][0]
		if isFinite(x) {
			if got != 0 {
				t.Fatal("finite overflow", x, got)
			}
		} else if isFinite(got) {
			t.Fatal("invalid hidden", x, got)
		}
	}
}

func TestTrainingForwardResultsWitnessChunking(t *testing.T) {
	for _, size := range []int{16384, 16385, 32769} {
		g := &transformerTapeGraph{constants: map[int][]float32{}, zeros: map[int]bool{}, params: map[string]int{}}
		values := make([]float32, size)
		for i := range values {
			values[i] = math.MaxFloat32
		}
		id := g.constant([]int{size}, values)
		witness := appendTrainingFiniteWitness(g, []int{id})
		g.tape.Outputs = []int{witness}
		p, err := materializeSequenceTape(g, nil, 64<<20)
		if err != nil {
			t.Fatal(size, err)
		}
		for _, c := range p.Tape.Commands {
			if c.Op == "linear-forward" && p.Tape.Tensors[c.Inputs[0]].Shape[1] > 16384 {
				t.Fatal("oversize witness")
			}
		}
		v := interpretFlowTape(t, p.Tape)
		if v[0][0] != 0 {
			t.Fatal("finite chunk overflow")
		}
	}
}
