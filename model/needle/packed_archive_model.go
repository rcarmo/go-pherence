package needle

import (
	"encoding/json"
	"fmt"
	"math"
	"slices"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	checkpoint "github.com/rcarmo/go-pherence/loader/needle"
)

// newPackedArchiveModel takes private parser-owned dense-required records.
// It does not route through newModel, which requires all decoded weights.
func newPackedArchiveModel(cp *checkpoint.Checkpoint, omitted map[string][]int, packed map[packedKey]*simd.CQMatrix) (*Model, error) {
	if cp == nil {
		return nil, fmt.Errorf("needle: nil packed archive checkpoint")
	}
	// Checkpoint construction already validated config, permutations and
	// complete layered CQ coverage; reject missing dense and unknown shapes.
	var c Config
	if err := json.Unmarshal(cp.Config, &c); err != nil {
		return nil, err
	}
	if err := c.validate(); err != nil {
		return nil, err
	}
	if c.Generation != 3 || !c.ArchiveDecoded {
		return nil, fmt.Errorf("needle: packed model requires deployed Needle3")
	}
	shapes := expectedShapes(c)
	for name, want := range shapes {
		if shape, ok := omitted[name]; ok {
			if !slices.Equal(shape, want) {
				return nil, fmt.Errorf("needle: invalid omitted shape %s", name)
			}
			if _, also := cp.Tensors[name]; also {
				return nil, fmt.Errorf("needle: mixed tensor %s", name)
			}
			continue
		}
		got, ok := cp.Tensors[name]
		if !ok || !slices.Equal(got.Shape, want) {
			return nil, fmt.Errorf("needle: missing/invalid dense tensor %s", name)
		}
	}
	for name := range omitted {
		if _, ok := shapes[name]; !ok {
			return nil, fmt.Errorf("needle: unknown omitted tensor %s", name)
		}
	}
	if err := validateAB(cp.Tensors); err != nil {
		return nil, err
	}
	m := &Model{config: c, rawConfig: append([]byte(nil), cp.Config...), deployed: true, archiveWindow: c.ArchiveKVWindow, compactPacked: true, tensors: make(map[string]checkpoint.Tensor, len(cp.Tensors)), packedShapes: make(map[string][]int, len(omitted)), packed: packed}
	var total int64
	for name, tensor := range cp.Tensors {
		if len(tensor.Shape) > 8 {
			return nil, fmt.Errorf("needle: excessive tensor rank %s", name)
		}
		n := 1
		for _, dim := range tensor.Shape {
			if dim <= 0 || n > (1<<29)/dim {
				return nil, fmt.Errorf("needle: excessive tensor %s", name)
			}
			n *= dim
		}
		total += int64(n)
		if total > 1<<29 || len(tensor.Data) != n {
			return nil, fmt.Errorf("needle: invalid tensor storage %s", name)
		}
		for _, v := range tensor.Data {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return nil, fmt.Errorf("needle: nonfinite tensor %s", name)
			}
		}
		m.tensors[name] = checkpoint.Tensor{Shape: slices.Clone(tensor.Shape), Data: tensor.Data}
	}
	for name, shape := range omitted {
		m.packedShapes[name] = slices.Clone(shape)
		rows, cols, layers, ok := packedPlanGeometry(c, name, shape)
		if !ok || isPackedPlanHead(name) {
			return nil, fmt.Errorf("needle: invalid omitted packed consumer %s", name)
		}
		for layer := 0; layer < layers; layer++ {
			keyLayer := layer
			if len(shape) == 2 {
				keyLayer = -1
			}
			p := packed[packedKey{name, keyLayer}]
			if p == nil || p.Rows() != rows || p.Cols() != cols {
				return nil, fmt.Errorf("needle: incomplete packed consumer %s layer %d", name, keyLayer)
			}
		}
	}
	m.p1 = numpyPermutation(padded(c.DModel), 11, len(c.LadderWidths) > 0)
	m.p2 = numpyPermutation(padded(c.DModel), 13, len(c.LadderWidths) > 0)
	return m, nil
}
