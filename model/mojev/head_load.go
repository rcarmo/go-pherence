package mojev

import (
	"fmt"

	"github.com/rcarmo/go-pherence/loader/weights"
)

// LoadHead loads only the four F32 PackedScorer head tensors. It never loads
// the encoder. The caller owns and closes the source; NewHeadWeights copies
// the decoded values so the returned head survives source closure.
// Failure at any tensor leaves no partially usable head.
func LoadHead(src weights.Source, width, rank int) (*HeadWeights, error) {
	if src == nil || width <= 0 || width > 4096 || rank <= 0 || rank > 4096 {
		return nil, fmt.Errorf("mojev: invalid head source or dimensions")
	}
	get := func(name string, shape []int) ([]float32, error) {
		data, dims, err := src.GetFloat32(name)
		if err != nil {
			return nil, fmt.Errorf("mojev: read %s: %w", name, err)
		}
		if len(dims) != len(shape) {
			return nil, fmt.Errorf("mojev: %s rank mismatch", name)
		}
		for i, want := range shape {
			if dims[i] != want {
				return nil, fmt.Errorf("mojev: %s dimension %d: got %d want %d", name, i, dims[i], want)
			}
		}
		return data, nil
	}
	gamma, err := get("norm.weight", []int{width})
	if err != nil {
		return nil, err
	}
	beta, err := get("norm.bias", []int{width})
	if err != nil {
		return nil, err
	}
	context, err := get("context_proj.weight", []int{rank, width})
	if err != nil {
		return nil, err
	}
	option, err := get("option_proj.weight", []int{rank, width})
	if err != nil {
		return nil, err
	}
	return NewHeadWeights(width, rank, gamma, beta, context, option)
}
