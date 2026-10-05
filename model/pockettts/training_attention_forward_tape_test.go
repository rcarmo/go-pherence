package pockettts

import (
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
	"testing"
)

func TestTrainingAttentionForwardTapeTransformerConvention(t *testing.T) {
	for _, rows := range []int{1, 3, 13} {
		for _, contextRows := range []int{0, 1, 2, rows} {
			if contextRows > rows {
				continue
			}
			m := tinyFlowLMTraining().Transformer
			layer := m.Layers[0]
			width, heads, dim := m.Width, m.Heads, m.HeadDim
			input := make([]float32, rows*width)
			for i := range input {
				input[i] = float32(math.Sin(float64(i) * .7))
			}
			cpu, _ := layer.forwardTraining(rows, width, heads, dim, contextRows, m.MaxPeriod, input)
			_, normed := transformerNormForward(rows, width, input, layer.Norm1Weight, layer.Norm1Bias, 1e-5)
			qkv := linearForwardRowsTraining(layer.InProjection, normed, rows)
			q, k, v := make([]float32, rows*width), make([]float32, rows*width), make([]float32, rows*width)
			for row := 0; row < rows; row++ {
				copy(q[row*width:(row+1)*width], qkv[row*width*3:row*width*3+width])
				copy(k[row*width:(row+1)*width], qkv[row*width*3+width:row*width*3+2*width])
				copy(v[row*width:(row+1)*width], qkv[row*width*3+2*width:(row+1)*width*3])
			}
			table := make([]float32, rows*dim)
			if err := train.RoPETableInto(table, rows, dim, 0, m.MaxPeriod); err != nil {
				t.Fatal(err)
			}
			tensor := []int{rows, heads, dim}
			dense := []int{heads, rows, rows}
			scale := float32(1 / math.Sqrt(float64(dim)))
			tape := train.Tape{Tensors: []train.TensorSpec{{Shape: tensor, Values: q}, {Shape: tensor, Values: k}, {Shape: tensor, Values: v}, {Shape: []int{rows, dim / 2, 2}, Values: table}, {Shape: tensor}, {Shape: tensor}, {Shape: dense, Values: make([]float32, heads*rows*rows)}, {Shape: dense}, {Shape: dense}, {Shape: tensor}}, Commands: []train.Command{
				{Op: "rope", Inputs: []int{0, 3}, Output: 4}, {Op: "rope", Inputs: []int{1, 3}, Output: 5},
				{Op: "attention-forward", Inputs: []int{4, 5, 2, 6}, Output: 7, Offset: contextRows, Scale: scale},
				{Op: "attention-forward", Inputs: []int{4, 5, 2, 7}, Output: 8, Offset: contextRows, Mode: 1, Scale: scale},
				{Op: "attention-forward", Inputs: []int{4, 5, 2, 8}, Output: 9, Offset: contextRows, Mode: 2, Scale: scale},
			}, Outputs: []int{4, 5, 8, 9}}
			got := interpretFlowTape(t, tape)
			assertSliceClose(t, "RoPE attention q", got[0], cpu.q, 1e-6)
			assertSliceClose(t, "RoPE attention k", got[1], cpu.k, 1e-6)
			assertSliceClose(t, "attention probabilities", got[2], cpu.probabilities, 2e-6)
			assertSliceClose(t, "attention outputs", got[3], cpu.attention, 2e-6)
		}
	}
}
