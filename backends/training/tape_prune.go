package training

import "fmt"

// PruneTape keeps the backwards dependency closure of declared roots. The full
// source tape is admitted first: a malformed dead command or over-budget source
// never disappears through pruning. Retained commands and operands keep their
// original order, modes and arithmetic; no constant folding or buffer reuse.
// Values are borrowed read-only; callers must keep source constants immutable.
// oldToNew maps live tensor IDs, with -1 for dead IDs. Constant-only roots
// produce a validated zero-command tape: upload/download but no dispatch.
func PruneTape(t Tape, roots []int, maxBytes int64) (Tape, []int, error) {
	if err := t.Validate(maxBytes); err != nil {
		return Tape{}, nil, err
	}
	if len(roots) == 0 {
		return Tape{}, nil, fmt.Errorf("tape pruning requires roots")
	}
	declared := make([]bool, len(t.Tensors))
	for _, id := range t.Outputs {
		declared[id] = true
	}
	live := make([]bool, len(t.Tensors))
	for _, id := range roots {
		if id < 0 || id >= len(live) || !declared[id] {
			return Tape{}, nil, fmt.Errorf("pruning root must be declared output")
		}
		live[id] = true
	}
	keep := make([]bool, len(t.Commands))
	retainedCommands := 0
	metadataInts := 0
	for i := len(t.Commands) - 1; i >= 0; i-- {
		c := t.Commands[i]
		if live[c.Output] {
			keep[i] = true
			retainedCommands++
			metadataInts += len(c.Inputs)
			for _, id := range c.Inputs {
				live[id] = true
			}
		}
	}
	mapping := make([]int, len(live))
	retainedTensors := 0
	for i := range mapping {
		mapping[i] = -1
		if live[i] {
			retainedTensors++
			metadataInts += len(t.Tensors[i].Shape)
		}
	}
	// The admitted dependency closure is now known. Reserve exactly its result
	// metadata once; source shapes/inputs still receive independent owned copies.
	// Constants retain their existing read-only borrowing contract.
	result := Tape{Tensors: make([]TensorSpec, 0, retainedTensors), Outputs: make([]int, 0, len(roots))}
	if retainedCommands > 0 {
		result.Commands = make([]Command, 0, retainedCommands)
	} // retain nil for constant-only graphs
	metadata := pruneMetadataArena{unassigned: metadataInts}
	for id, s := range t.Tensors {
		if live[id] {
			mapping[id] = len(result.Tensors)
			shape := metadata.take(len(s.Shape))
			copy(shape, s.Shape)
			result.Tensors = append(result.Tensors, TensorSpec{Shape: shape, Values: s.Values, External: s.External})
		}
	}
	for i, c := range t.Commands {
		if keep[i] {
			c.Output = mapping[c.Output]
			inputs := metadata.take(len(c.Inputs))
			for j, id := range c.Inputs {
				inputs[j] = mapping[id]
			}
			c.Inputs = inputs
			result.Commands = append(result.Commands, c)
		}
	}
	for _, id := range roots {
		result.Outputs = append(result.Outputs, mapping[id])
	}
	if err := result.Validate(maxBytes); err != nil {
		return Tape{}, nil, err
	}
	return result, mapping, nil
}

// Per-call chunked integer metadata. Disjoint full-slice expressions preserve
// mutation and append isolation. Small chunks bound incidental retention when
// a consumer keeps one shape/operand slice after dropping the rest of the tape.
// Source validation bounds shape ranks and supported command input counts.
type pruneMetadataArena struct {
	remaining  []int
	unassigned int
}

func (a *pruneMetadataArena) take(n int) []int {
	if len(a.remaining) < n {
		// Complete source validation bounds totals and each slice length.
		// Smaller final chunks avoid padding every tiny constant/utility tape.
		a.remaining = make([]int, max(n, min(256, a.unassigned)))
	}
	out := a.remaining[:n:n]
	a.remaining = a.remaining[n:]
	a.unassigned -= n
	return out
}
