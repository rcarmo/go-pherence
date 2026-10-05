package pockettts

import (
	train "github.com/rcarmo/go-pherence/backends/training"
	"sort"
)

func (b *ResidentTrainingBackend) embeddingCompactEnabled() bool {
	return b != nil && (b.diagonalAdmissionEnabled() || b.Scope == "ptx-resident-embeddingcompact-hybrid-v1" || b.Scope == "vulkan-resident-embeddingcompact-hybrid-v1")
}

// Admit the complete dense source before adding compact gradient/index roots.
// Keep all original IDs, commands and outputs for public downloads and full
// logical admission. Pruning removes dense VJP only if no caller requests it.
func compactResidentEmbeddingTape(source train.Tape, bindings map[string]train.ResidentTapeBinding, maxBytes int64) (train.Tape, map[string]train.ResidentTapeBinding, map[string]train.ResidentCompactEmbeddingBinding, error) {
	if err := train.ValidateResidentTapeBindings(source, bindings, maxBytes); err != nil {
		return train.Tape{}, nil, nil, err
	}
	// Sorted binding names make new IDs and command order deterministic.
	names := make([]string, 0, len(bindings))
	for name := range bindings {
		names = append(names, name)
	}
	sort.Strings(names)
	t := source
	var owned map[string]train.ResidentTapeBinding
	var compact map[string]train.ResidentCompactEmbeddingBinding
	for _, name := range names {
		binding := bindings[name]
		for _, command := range source.Commands {
			if command.Output != binding.Gradient || command.Op != "embedding" || command.Mode != 1 {
				continue
			}
			if compact == nil {
				compact = map[string]train.ResidentCompactEmbeddingBinding{}
				owned = make(map[string]train.ResidentTapeBinding, len(bindings))
				for k, v := range bindings {
					owned[k] = v
				}
				t.Tensors = append([]train.TensorSpec(nil), source.Tensors...)
				t.Commands = append([]train.Command(nil), source.Commands...)
				t.Outputs = append([]int(nil), source.Outputs...)
			}
			index, err := train.BuildCompactEmbeddingIndex(source.Tensors[command.Inputs[1]].Values, command.Parts)
			if err != nil {
				return train.Tape{}, nil, nil, err
			}
			indexID := len(t.Tensors)
			gradientID := indexID + 1
			t.Tensors = append(t.Tensors, train.TensorSpec{Shape: []int{len(index)}, Values: index}, train.TensorSpec{Shape: []int{command.Rows * command.Width}})
			command.Inputs = append([]int(nil), command.Inputs...)
			command.Inputs[1] = indexID
			command.Output = gradientID
			command.Mode = 3
			t.Commands = append(t.Commands, command)
			t.Outputs = append(t.Outputs, indexID, gradientID)
			owned[name] = train.ResidentTapeBinding{Parameter: binding.Parameter, Gradient: gradientID}
			compact[name] = train.ResidentCompactEmbeddingBinding{Index: indexID, Rows: command.Rows, Width: command.Width, Vocabulary: command.Parts}
		}
	}
	if compact == nil {
		return source, bindings, nil, nil
	}
	if err := train.ValidateResidentCompactTapeBindings(t, owned, compact, maxBytes); err != nil {
		return train.Tape{}, nil, nil, err
	}
	return t, owned, compact, nil
}
