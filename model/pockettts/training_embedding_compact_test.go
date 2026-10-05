package pockettts

import (
	train "github.com/rcarmo/go-pherence/backends/training"
	"testing"
)

func TestTrainingCompactEmbeddingTapeParity(t *testing.T) {
	for _, ids := range [][]float32{{0}, {2, 0, 2, 3, 0}, {1, 1, 1, 1}} {
		const width, vocabulary = 3, 5
		rows := len(ids)
		source := make([]float32, rows*width)
		for i := range source {
			source[i] = float32((i*7)%13-6) * .00137
		}
		index, err := train.BuildCompactEmbeddingIndex(ids, vocabulary)
		if err != nil {
			t.Fatal(err)
		}
		makeTape := func(values []float32, mode uint32, n int) train.Tape {
			return train.Tape{Tensors: []train.TensorSpec{{Shape: []int{len(source)}, Values: append([]float32(nil), source...)}, {Shape: []int{len(values)}, Values: append([]float32(nil), values...)}, {Shape: []int{n}}}, Commands: []train.Command{{Op: "embedding", Inputs: []int{0, 1}, Output: 2, Rows: rows, Width: width, Parts: vocabulary, Mode: mode}}, Outputs: []int{2}}
		}
		legacy, compact := makeTape(ids, 1, vocabulary*width), makeTape(index, 3, rows*width)
		for _, tape := range []train.Tape{legacy, compact} {
			if err := tape.Validate(1024); err != nil {
				t.Fatal(err)
			}
		}
		expected := interpretFlowTape(t, legacy)[0]
		got := interpretFlowTape(t, compact)[0]
		dense := make([]float32, len(expected))
		for slot := 0; slot < rows; slot++ {
			token := int(index[slot])
			if token < 0 {
				if !zeroTrainingSeed(got[slot*width : (slot+1)*width]) {
					t.Fatal("compact padding")
				}
				continue
			}
			copy(dense[token*width:(token+1)*width], got[slot*width:(slot+1)*width])
		}
		assertSliceClose(t, "compact versus sequence-order dense VJP", dense, expected, 2e-6)
		pruned, _, err := train.PruneTape(compact, compact.Outputs, 1024)
		if err != nil {
			t.Fatal(err)
		}
		assertSliceClose(t, "pruned compact VJP", interpretFlowTape(t, pruned)[0], got, 2e-6)
	}
}
