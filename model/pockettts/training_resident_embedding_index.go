package pockettts

import train "github.com/rcarmo/go-pherence/backends/training"

func (b *ResidentTrainingBackend) embeddingIndexEnabled() bool {
	return b != nil && (b.Scope == "ptx-resident-embeddingindex-hybrid-v1" || b.Scope == "vulkan-resident-embeddingindex-hybrid-v1")
}

// Rewrite only after complete source/binding admission. Dense gradient/output
// IDs and command order stay fixed; owned constant indices are appended and
// only the changed command headers/operand slices acquire new ownership.
func indexResidentEmbeddingTape(t train.Tape, bindings map[string]train.ResidentTapeBinding, maxBytes int64) (train.Tape, error) {
	if err := train.ValidateResidentTapeBindings(t, bindings, maxBytes); err != nil {
		return train.Tape{}, err
	}
	count := 0
	for _, c := range t.Commands {
		if c.Op == "embedding" && c.Mode == 1 {
			count++
		}
	}
	if count == 0 {
		return t, nil
	}
	source := t
	t.Tensors = make([]train.TensorSpec, len(source.Tensors), len(source.Tensors)+count)
	copy(t.Tensors, source.Tensors)
	t.Commands = append([]train.Command(nil), source.Commands...)
	for i, c := range source.Commands {
		if c.Op != "embedding" || c.Mode != 1 {
			continue
		}
		index, err := train.BuildEmbeddingIndex(source.Tensors[c.Inputs[1]].Values, c.Parts)
		if err != nil {
			return train.Tape{}, err
		}
		id := len(t.Tensors)
		t.Tensors = append(t.Tensors, train.TensorSpec{Shape: []int{len(index)}, Values: index})
		c.Inputs = append([]int(nil), c.Inputs...)
		c.Inputs[1] = id
		c.Mode = 2
		t.Commands[i] = c
	}
	// Retain old IDs/constants even if dead. Combined admission must succeed
	// before root pruning or any native allocation/gradient accumulation.
	if err := train.ValidateResidentTapeBindings(t, bindings, maxBytes); err != nil {
		return train.Tape{}, err
	}
	return t, nil
}
