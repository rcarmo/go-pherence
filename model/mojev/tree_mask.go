package mojev

import "fmt"

// TreeMask builds a bounded reference visibility matrix for one packed row.
// State may see state; a question may see state and itself; a candidate may
// see state, its own question and itself. Padding keeps only its diagonal.
// This is model-free and does not implement the Qwen3.5 attention backend.
func TreeMask(state []bool, questions [][]bool, candidates [][][]bool) ([][]bool, error) {
	length := len(state)
	if length == 0 || length > 4096 || len(questions) == 0 || len(questions) > 256 || len(candidates) != len(questions) {
		return nil, fmt.Errorf("mojev: invalid tree-mask geometry")
	}
	// Each position has at most one owner. Empty spans represent absent options;
	// overlapping spans would silently break the isolation contract.
	type owner struct{ kind, field, option int } // 0 padding, 1 state, 2 question, 3 candidate
	owners := make([]owner, length)
	for pos, active := range state {
		if active {
			owners[pos].kind = 1
		}
	}
	set := func(span []bool, value owner) error {
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
		if err := set(q, owner{kind: 2, field: f}); err != nil {
			return nil, err
		}
		for n, c := range candidates[f] {
			if err := set(c, owner{kind: 3, field: f, option: n}); err != nil {
				return nil, err
			}
		}
	}
	allowed := make([][]bool, length)
	for row, from := range owners {
		allowed[row] = make([]bool, length)
		for column, to := range owners {
			allowed[row][column] = row == column ||
				(from.kind != 0 && to.kind == 1) ||
				(from.kind == 2 && to.kind == 2 && from.field == to.field) ||
				(from.kind == 3 && to.kind == 2 && from.field == to.field) ||
				(from.kind == 3 && to.kind == 3 && from.field == to.field && from.option == to.option)
		}
	}
	return allowed, nil
}
