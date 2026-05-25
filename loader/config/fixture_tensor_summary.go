package config

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
)

// FixtureTensorSummary is a compact, JSON-friendly tensor summary used by
// metadata/parity fixture scripts. Float32 hashes are over little-endian
// IEEE-754 bytes so Python, Go, and other runtimes can compare outputs without
// storing full tensors in git.
type FixtureTensorSummary struct {
	Name        string    `json:"name"`
	DType       string    `json:"dtype"`
	Shape       []int     `json:"shape"`
	SHA256LEF32 string    `json:"sha256_le_f32"`
	Min         float32   `json:"min"`
	Max         float32   `json:"max"`
	Mean        float32   `json:"mean"`
	FirstValues []float32 `json:"first_values"`
}

func SummarizeFixtureFloat32Tensor(name string, shape []int, values []float32) (FixtureTensorSummary, error) {
	n, err := FixtureShapeNumel(shape)
	if err != nil {
		return FixtureTensorSummary{}, err
	}
	if n != len(values) {
		return FixtureTensorSummary{}, fmt.Errorf("fixture tensor summary %q: shape %v has %d elements, got %d", name, shape, n, len(values))
	}
	if len(values) == 0 {
		return FixtureTensorSummary{}, fmt.Errorf("fixture tensor summary %q: empty tensor", name)
	}
	raw := make([]byte, len(values)*4)
	minV, maxV := float32(math.Inf(1)), float32(math.Inf(-1))
	var sum float64
	for i, v := range values {
		binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(v))
		if v < minV {
			minV = v
		}
		if v > maxV {
			maxV = v
		}
		sum += float64(v)
	}
	first := len(values)
	if first > 16 {
		first = 16
	}
	return FixtureTensorSummary{
		Name:        name,
		DType:       "float32",
		Shape:       append([]int(nil), shape...),
		SHA256LEF32: fmt.Sprintf("%x", sha256.Sum256(raw)),
		Min:         minV,
		Max:         maxV,
		Mean:        float32(sum / float64(len(values))),
		FirstValues: append([]float32(nil), values[:first]...),
	}, nil
}

func FixtureShapeNumel(shape []int) (int, error) {
	if len(shape) == 0 {
		return 0, fmt.Errorf("fixture tensor shape: empty")
	}
	n := 1
	maxInt := int(^uint(0) >> 1)
	for _, d := range shape {
		if d <= 0 {
			return 0, fmt.Errorf("fixture tensor shape: invalid dimension %d in %v", d, shape)
		}
		if n > maxInt/d {
			return 0, fmt.Errorf("fixture tensor shape: overflow for %v", shape)
		}
		n *= d
	}
	return n, nil
}
