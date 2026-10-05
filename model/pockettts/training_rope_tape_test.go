package pockettts

import (
	train "github.com/rcarmo/go-pherence/backends/training"
	"testing"
)

// Compare the new host tape schedule against the existing transformer scalar
// convention without changing the CPU path or introducing device execution.
func TestTrainingRoPETapeTransformerConvention(t *testing.T) {
	for _, v := range []struct {
		rows, heads, dim, origin int
		period                   float64
	}{{1, 1, 2, 0, 10000}, {13, 3, 14, 137, 10000}, {2, 2, 256, 16384, 4096}} {
		count := v.rows * v.heads * v.dim
		table := make([]float32, v.rows*v.dim)
		if err := train.RoPETableInto(table, v.rows, v.dim, v.origin, v.period); err != nil {
			t.Fatal(err)
		}
		x := make([]float32, count)
		for i := range x {
			x[i] = float32(i%17-8) * .1
		}
		for mode := uint32(0); mode < 2; mode++ {
			expected := append([]float32(nil), x...)
			for row := 0; row < v.rows; row++ {
				ropeScalar(expected[row*v.heads*v.dim:(row+1)*v.heads*v.dim], v.origin+row, v.heads, v.dim, v.period, mode == 1)
			}
			tape := train.Tape{Tensors: []train.TensorSpec{{Shape: []int{v.rows, v.heads, v.dim}, Values: x}, {Shape: []int{v.rows, v.dim / 2, 2}, Values: table}, {Shape: []int{v.rows, v.heads, v.dim}}}, Commands: []train.Command{{Op: "rope", Inputs: []int{0, 1}, Output: 2, Mode: mode}}, Outputs: []int{2}}
			got := interpretFlowTape(t, tape)[0]
			assertSliceClose(t, "RoPE transformer convention", got, expected, 1e-6)
		}
	}
}
