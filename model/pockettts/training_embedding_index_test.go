package pockettts

import (
	train "github.com/rcarmo/go-pherence/backends/training"
	"testing"
)

func TestTrainingEmbeddingIndexedTapeParity(t *testing.T) {
	for _, ids := range [][]float32{{0}, {2, 0, 2, 3, 0}, {1, 1, 1, 1}} {
		const width, vocabulary = 3, 5
		source := make([]float32, len(ids)*width)
		for i := range source {
			source[i] = float32((i*7)%13-6) * .03125
		}
		index, err := train.BuildEmbeddingIndex(ids, vocabulary)
		if err != nil {
			t.Fatal(err)
		}
		makeTape := func(values []float32, mode uint32) train.Tape {
			return train.Tape{Tensors: []train.TensorSpec{{Shape: []int{len(source)}, Values: append([]float32(nil), source...)}, {Shape: []int{len(values)}, Values: append([]float32(nil), values...)}, {Shape: []int{vocabulary * width}}}, Commands: []train.Command{{Op: "embedding", Inputs: []int{0, 1}, Output: 2, Rows: len(ids), Width: width, Parts: vocabulary, Mode: mode}}, Outputs: []int{2}}
		}
		legacy, indexed := makeTape(ids, 1), makeTape(index, 2)
		for _, tape := range []train.Tape{legacy, indexed} {
			if err := tape.Validate(1024); err != nil {
				t.Fatal(err)
			}
		}
		expected := interpretFlowTape(t, legacy)[0]
		got := interpretFlowTape(t, indexed)[0]
		assertSliceClose(t, "indexed versus sequence-order dense VJP", got, expected, 2e-6)
		pruned, _, err := train.PruneTape(indexed, indexed.Outputs, 1024)
		if err != nil {
			t.Fatal(err)
		}
		assertSliceClose(t, "pruned indexed VJP", interpretFlowTape(t, pruned)[0], expected, 2e-6)
	}
}
