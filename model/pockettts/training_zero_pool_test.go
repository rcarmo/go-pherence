package pockettts

import (
	train "github.com/rcarmo/go-pherence/backends/training"
	"testing"
)

func TestTrainingZeroPoolOwnership(t *testing.T) {
	var a, b trainingZeroPool
	x, y, z := a.values(256), a.values(256), a.values(257)
	if &x[0] != &y[0] || f32SlicesOverlap(x, z) || f32SlicesOverlap(x, b.values(256)) {
		t.Fatal("extent/graph pool isolation")
	}
	if !zeroTrainingSeed(x) || !zeroTrainingSeed(z) {
		t.Fatal("not zero")
	}
	// Read-only contract during execution; mutation here only demonstrates that a
	// new graph does not reuse old storage, not permission to mutate live pools.
	x[0] = 9
	if b.values(256)[0] != 0 {
		t.Fatal("crossgraph contamination")
	}
	g := &transformerTapeGraph{borrowParameters: true}
	if !f32SlicesOverlap(g.zeroValues(16), g.zeroValues(16)) {
		t.Fatal("resident zeros not pooled")
	}
	public := &transformerTapeGraph{}
	if f32SlicesOverlap(public.zeroValues(16), public.zeroValues(16)) {
		t.Fatal("public zero ownership changed")
	}
	fg := &flowTapeGraph{borrowParameters: true}
	u, v := fg.zero(16), fg.zero(16)
	if u == v || !f32SlicesOverlap(fg.tape.Tensors[u].Values, fg.tape.Tensors[v].Values) {
		t.Fatal("flow zero ID/storage")
	}
	pf := &flowTapeGraph{}
	u, v = pf.zero(16), pf.zero(16)
	if f32SlicesOverlap(pf.tape.Tensors[u].Values, pf.tape.Tensors[v].Values) {
		t.Fatal("public flow zero ownership")
	}
}
func TestTrainingZeroPoolProjectionContract(t *testing.T) {
	l := LinearF32{In: 2, Out: 2, Weight: []float32{1, 2, 3, 4}, Bias: []float32{.1, .2}}
	input, seed := []float32{0, 0, 0, 0}, []float32{0, 0, 0, 0}
	p, err := buildProjectionTrainingTape(l, "w", "b", input, seed, 2, 1<<20, true)
	if err != nil {
		t.Fatal(err)
	}
	owned, err := BuildProjectionTrainingTape(l, "w", "b", input, seed, 2, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	count := 0
	shared := false
	for i, x := range p.Tape.Tensors {
		if x.Values == nil || !zeroTrainingSeed(x.Values) {
			continue
		}
		count++
		if f32SlicesOverlap(x.Values, input) || f32SlicesOverlap(x.Values, seed) {
			t.Fatal("caller zero borrowed")
		}
		for j := 0; j < i; j++ {
			y := p.Tape.Tensors[j]
			if y.Values != nil && f32SlicesOverlap(x.Values, y.Values) {
				shared = true
			}
		}
	}
	// Construction may now intern a repeated immutable zero as one declared
	// source ID. Otherwise distinct zero views must share private storage.
	if count == 0 {
		t.Fatal("missing literal zero source")
	}
	// Equal-shape tangent placeholders are now the same source ID.
	forward := p.Tape.Commands[0]
	backward := p.Tape.Commands[1]
	if !shared && (forward.Inputs[1] != backward.Inputs[4]) {
		t.Fatal("repeated zero extent not interned or pooled")
	}
	before := interpretFlowTape(t, p.Tape)
	input[0] = 99
	seed[0] = 99
	after := interpretFlowTape(t, p.Tape)
	for i, x := range before {
		assertSliceClose(t, "caller zeros copied", after[i], x, 0)
	}
	for i, x := range owned.Tape.Tensors {
		if x.Values == nil {
			continue
		}
		for j := 0; j < i; j++ {
			if f32SlicesOverlap(x.Values, owned.Tape.Tensors[j].Values) {
				t.Fatal("public distinct constant storage")
			}
		}
	}
	// Shared source arrays do not permit initial-value writes in either mode.
	bad := p.Tape
	bad.Commands = append([]train.Command(nil), p.Tape.Commands...)
	bad.Commands[0].Output = 0
	if bad.Validate(1<<20) == nil {
		t.Fatal("zero source became mutable destination")
	}
	if _, err = buildProjectionTrainingTape(l, "w", "b", input, seed, 2, 1, true); err == nil {
		t.Fatal("logical ceiling bypassed")
	}
}
func TestTrainingZeroPoolConcurrentGraphs(t *testing.T) {
	for i := 0; i < 8; i++ {
		t.Run(string(rune('a'+i)), func(t *testing.T) {
			t.Parallel()
			l := LinearF32{In: 2, Out: 2, Weight: []float32{1, 2, 3, 4}, Bias: []float32{.1, .2}}
			var previous []float32
			for k := 0; k < 10; k++ {
				p, err := buildProjectionTrainingTape(l, "w", "b", []float32{.2, .3}, []float32{.4, .5}, 1, 1<<20, true)
				if err != nil {
					t.Fatal(err)
				}
				// Input tangent zero is the second operand of the forward command,
				// never a caller seed or result. Avoid fixture-specific tensor IDs.
				x := p.Tape.Tensors[p.Tape.Commands[0].Inputs[1]].Values
				if !zeroTrainingSeed(x) || f32SlicesOverlap(x, previous) {
					t.Fatal("graph-local lifetime")
				}
				previous = x
				out := interpretFlowTape(t, p.Tape)
				if len(out) != 4 {
					t.Fatal("output")
				}
			}
		})
	}
}
func TestTrainingZeroPoolLogicalAdmission(t *testing.T) {
	m := weightTapeFixture()
	p, err := m.buildWeightTrainingTape([]float32{.2, .3, .4, .5}, []float32{.3, -.4}, 2, 1<<20, nil, true)
	if err != nil {
		t.Fatal(err)
	}
	var logical int64
	for _, s := range p.Tape.Tensors {
		n, _ := s.Elements()
		logical += int64(n) * 4
	}
	if _, err = m.buildWeightTrainingTape([]float32{.2, .3, .4, .5}, []float32{.3, -.4}, 2, logical-1, nil, true); err == nil {
		t.Fatal("pool lowered logical admission")
	}
	if _, err = m.buildWeightTrainingTape([]float32{.2, .3, .4, .5}, []float32{.3, -.4}, 2, logical, nil, true); err != nil {
		t.Fatal("exact ceiling", err)
	}
}
func BenchmarkTrainingZeroPoolMaterialization(b *testing.B) {
	for _, mode := range []string{"independent", "graph-local-pool"} {
		b.Run(mode, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				g := &transformerTapeGraph{borrowParameters: mode == "graph-local-pool"}
				arrays := make([][]float32, 64)
				for k := range arrays {
					arrays[k] = g.zeroValues(32768)
				}
				if arrays[63][32767] != 0 {
					b.Fatal("zero")
				}
			}
		})
	}
}
