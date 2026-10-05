package pockettts

import (
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
)

// TrainingBranchBackend evaluates explicit row-local interpolation/adjoint
// arithmetic. Boundary vectors are still host-owned between utility tapes.
// Kinds have fixed arity; no arbitrary graph or fallback is admitted.
type TrainingBranchBackend interface {
	EvaluateBranch(*FlowHeadCPU, string, [][]float32, float32) ([][]float32, error)
}

func (g *transformerTapeGraph) branchAdd(a, b int, scale float32) int {
	n, _ := g.tape.Tensors[a].Elements()
	id := g.command("vector", []int{a, b}, []int{n}, 0)
	g.tape.Commands[len(g.tape.Commands)-1].Scale = scale
	return id
}
func (g *transformerTapeGraph) branchScale(a int, scale float32) int {
	n, _ := g.tape.Tensors[a].Elements()
	return g.branchAdd(g.zero(n), a, scale)
}
func (g *transformerTapeGraph) branchDot(a, b int) int {
	n, _ := g.tape.Tensors[a].Elements()
	return g.reshape(g.command("linear-forward", []int{g.reshape(a, 1, n), g.zero(1, n), g.reshape(b, 1, n), g.zero(1)}, []int{1, 1}, 0), 1)
}

// BuildBranchTrainingTape preserves explicit F32 multiply/add boundaries using
// existing disjoint vector and affine primitives. No new native kernel needed.
func BuildBranchTrainingTape(kind string, inputs [][]float32, scalar float32, maxBytes int64) (train.Tape, error) {
	var empty train.Tape
	arity := map[string]int{"interpolate": 2, "advance": 3, "negate": 1, "mixed": 4, "finish": 6, "diagonal-target": 2, "scale": 1}
	count, ok := arity[kind]
	if !ok || len(inputs) != count || maxBytes <= 0 || !isFinite(scalar) {
		return empty, fmt.Errorf("branch kind/arity/scalar/ceiling")
	}
	n := len(inputs[0])
	if n < 1 || n > 16384 {
		return empty, fmt.Errorf("branch channel bound")
	}
	if (kind == "interpolate" || kind == "advance" || kind == "mixed" || kind == "finish" || kind == "diagonal-target") && (scalar < 0 || scalar > 1) {
		return empty, fmt.Errorf("branch time/delta bound")
	}
	for i, x := range inputs {
		want := n
		if kind == "finish" && i >= 4 {
			want = 2
			if i == 5 {
				want = 1
			}
		}
		if len(x) != want || !finiteF32(x) {
			return empty, fmt.Errorf("branch input extent/nonfinite")
		}
	}
	g := &transformerTapeGraph{constants: map[int][]float32{}, zeros: map[int]bool{}, params: map[string]int{}}
	ids := make([]int, count)
	for i, x := range inputs {
		ids[i] = g.constant([]int{len(x)}, x)
	}
	switch kind {
	case "interpolate":
		g.tape.Outputs = []int{g.branchAdd(g.branchScale(ids[0], scalar), g.branchScale(ids[1], 1-scalar), 1), g.branchAdd(ids[0], ids[1], -1)}
	case "advance":
		g.tape.Outputs = []int{g.branchAdd(ids[0], ids[1], scalar), g.branchAdd(ids[1], ids[2], scalar)}
	case "negate":
		g.tape.Outputs = []int{g.branchScale(ids[0], -1)}
	case "mixed":
		gv := g.branchAdd(ids[0], ids[1], scalar)
		gt := g.branchScale(ids[0], scalar)
		delta := g.branchAdd(g.branchDot(ids[0], ids[3]), g.branchDot(ids[1], ids[2]), 1)
		g.tape.Outputs = []int{gv, gt, delta}
	case "finish":
		dx := g.branchAdd(ids[0], ids[1], 1)
		target := g.branchScale(dx, scalar)
		noise := g.branchScale(dx, 1-scalar)
		ds := g.branchAdd(g.branchAdd(g.branchDot(dx, g.branchAdd(ids[2], ids[3], -1)), g.slice(ids[4], 0, 1), 1), ids[5], -1)
		dt := g.branchAdd(g.slice(ids[4], 1, 1), ids[5], 1)
		g.tape.Outputs = []int{target, noise, g.branchAdd(g.pad(ds, 0, 2), g.pad(dt, 1, 2), 1)}
	case "diagonal-target":
		g.tape.Outputs = []int{g.branchAdd(g.branchScale(ids[0], scalar), ids[1], -1)}
	case "scale":
		g.tape.Outputs = []int{g.branchScale(ids[0], scalar)}
	}
	result, err := materializeSequenceTape(g, nil, maxBytes)
	return result.Tape, err
}
func validateBranchBoundary(kind string, values, inputs [][]float32) error {
	if len(inputs) == 0 {
		return fmt.Errorf("branch boundary inputs")
	}
	n := len(inputs[0])
	shapes := map[string][]int{"interpolate": {n, n}, "advance": {n, n}, "negate": {n}, "mixed": {n, n, 1}, "finish": {n, n, 2}, "diagonal-target": {n}, "scale": {n}}
	want, ok := shapes[kind]
	if !ok || len(values) != len(want) {
		return fmt.Errorf("branch boundary count/kind")
	}
	for i, x := range values {
		if len(x) != want[i] || !finiteF32(x) {
			return fmt.Errorf("branch boundary extent/nonfinite")
		}
		for _, y := range inputs {
			if f32SlicesOverlap(x, y) {
				return fmt.Errorf("branch borrowed boundary")
			}
		}
		for _, y := range values[:i] {
			if f32SlicesOverlap(x, y) {
				return fmt.Errorf("branch outputs overlap")
			}
		}
	}
	return nil
}
func (b *ResidentTrainingBackend) ownsBranch() bool {
	return b != nil && (b.Scope == "ptx-resident-branch-hybrid-v1" || b.Scope == "vulkan-resident-branch-hybrid-v1" || b.ownsStepReduction())
}
func (b *ResidentTrainingBackend) branchOwnerMatches(t *FullTrainer) bool {
	if b.ownsBranch() {
		return t.Flow.BranchBackend == b
	}
	return t.Flow.BranchBackend == nil
}
func (b *ResidentTrainingBackend) EvaluateBranch(m *FlowHeadCPU, kind string, inputs [][]float32, scalar float32) ([][]float32, error) {
	if err := b.admitGraph(); err != nil {
		return nil, err
	}
	if !b.ownsBranch() || m != b.trainer.Flow {
		return nil, b.fail(fmt.Errorf("resident branch owner/scope changed"))
	}
	tape, err := BuildBranchTrainingTape(kind, inputs, scalar, b.MaxTapeBytes)
	if err != nil {
		return nil, b.fail(err)
	}
	values, err := b.runUtilityTape(tape, tape.Outputs)
	if err != nil {
		return nil, b.fail(err)
	}
	if err := validateBranchBoundary(kind, values, inputs); err != nil {
		return nil, b.fail(err)
	}
	return values, nil
}
func evaluateTrainingBranch(backend TrainingBranchBackend, m *FlowHeadCPU, kind string, inputs [][]float32, scalar float32) ([][]float32, error) {
	values, err := backend.EvaluateBranch(m, kind, inputs, scalar)
	if err != nil {
		return nil, err
	}
	if err := validateBranchBoundary(kind, values, inputs); err != nil {
		return nil, err
	}
	return values, nil
}
