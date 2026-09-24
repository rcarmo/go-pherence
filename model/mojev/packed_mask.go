package mojev

import (
	"fmt"
	"math"
)

// AdditiveTreeMask builds the single-row F32 attention mask after the packed
// token mask is applied. Allowed entries are zero; all others use the most
// negative finite F32 value. Every row retains its diagonal, including padding.
// The returned rows are request-owned, capacity-limited views of one matrix.
// This is a model-free reference, not an encoder attention implementation.
func AdditiveTreeMask(state []bool, questions [][]bool, candidates [][][]bool, packed []bool) ([][]float32, error) {
	if len(packed) != len(state) {
		return nil, fmt.Errorf("mojev: invalid packed-mask length")
	}
	owners, err := treeOwners(state, questions, candidates)
	if err != nil {
		return nil, err
	}
	const floorBits = uint32(0xff7fffff) // torch.finfo(torch.float32).min
	floor := math.Float32frombits(floorBits)
	length := len(owners)
	mask := make([][]float32, length)
	data := make([]float32, length*length)
	for row, from := range owners {
		mask[row] = data[row*length : (row+1)*length : (row+1)*length]
		for column, to := range owners {
			if row == column {
				continue
			}
			visible := false
			if packed[column] {
				switch from.kind {
				case 1:
					visible = to.kind == 1
				case 2:
					visible = to.kind == 1 || (to.kind == 2 && to.field == from.field)
				case 3:
					visible = to.kind == 1 || (to.field == from.field && (to.kind == 2 || (to.kind == 3 && to.option == from.option)))
				}
			}
			if !visible {
				mask[row][column] = floor
			}
		}
	}
	return mask, nil
}
