package mojev

import "fmt"

// treeOwner encodes a single token's branch. A zero kind is unclaimed padding.
type treeOwner struct{ kind, field, option int } // 1 state, 2 question, 3 candidate

// treeOwners validates the single-owner rule before any output is allocated.
func treeOwners(state []bool, questions [][]bool, candidates [][][]bool) ([]treeOwner, error) {
	length := len(state)
	if length == 0 || length > 4096 || len(questions) == 0 || len(questions) > 256 || len(candidates) != len(questions) {
		return nil, fmt.Errorf("mojev: invalid tree-mask geometry")
	}
	// Each position has at most one owner. Empty spans represent absent options;
	// overlapping spans would silently break the isolation contract.
	owners := make([]treeOwner, length)
	for pos, active := range state {
		if active {
			owners[pos].kind = 1
		}
	}
	set := func(span []bool, value treeOwner) error {
		if len(span) != length {
			return fmt.Errorf("mojev: invalid span length")
		}
		for pos, active := range span {
			if !active {
				continue
			}
			if owners[pos].kind != 0 {
				return fmt.Errorf("mojev: overlapping spans")
			}
			owners[pos] = value
		}
		return nil
	}
	for f, q := range questions {
		if len(candidates[f]) == 0 || len(candidates[f]) > 64 {
			return nil, fmt.Errorf("mojev: invalid candidate count")
		}
		if err := set(q, treeOwner{kind: 2, field: f}); err != nil {
			return nil, err
		}
		for n, c := range candidates[f] {
			if err := set(c, treeOwner{kind: 3, field: f, option: n}); err != nil {
				return nil, err
			}
		}
	}
	return owners, nil
}

// TreeMask builds a bounded reference visibility matrix for one packed row.
// State may see state; a question may see state and itself; a candidate may
// see state, its own question and itself. Padding keeps only its diagonal.
// Rows are capacity-limited views of one owned matrix. This is model-free
// and does not implement the Qwen3.5 attention backend.
func TreeMask(state []bool, questions [][]bool, candidates [][][]bool) ([][]bool, error) {
	owners, err := treeOwners(state, questions, candidates)
	if err != nil {
		return nil, err
	}
	length := len(owners)
	allowed := make([][]bool, length)
	data := make([]bool, length*length)
	for row, from := range owners {
		allowed[row] = data[row*length : (row+1)*length : (row+1)*length]
		for column, to := range owners {
			if row == column {
				allowed[row][column] = true
				continue
			}
			switch from.kind {
			case 1:
				allowed[row][column] = to.kind == 1
			case 2:
				allowed[row][column] = to.kind == 1 || (to.kind == 2 && to.field == from.field)
			case 3:
				allowed[row][column] = to.kind == 1 || (to.field == from.field && (to.kind == 2 || (to.kind == 3 && to.option == from.option)))
			}
		}
	}
	return allowed, nil
}
