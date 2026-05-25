package hunyuan3d

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
)

// TensorSummary mirrors the compact fixture summaries emitted by the Python
// Hunyuan3D fixture scripts. Float32 hashes use little-endian IEEE-754 bytes.
type TensorSummary struct {
	Name        string    `json:"name"`
	DType       string    `json:"dtype"`
	Shape       []int     `json:"shape"`
	SHA256LEF32 string    `json:"sha256_le_f32"`
	Min         float32   `json:"min"`
	Max         float32   `json:"max"`
	Mean        float32   `json:"mean"`
	FirstValues []float32 `json:"first_values"`
}

func SummarizeFloat32Tensor(name string, shape []int, values []float32) (TensorSummary, error) {
	n, err := shapeNumel(shape)
	if err != nil {
		return TensorSummary{}, err
	}
	if n != len(values) {
		return TensorSummary{}, fmt.Errorf("hunyuan3d tensor summary %q: shape %v has %d elements, got %d", name, shape, n, len(values))
	}
	raw := make([]byte, len(values)*4)
	if len(values) == 0 {
		return TensorSummary{}, fmt.Errorf("hunyuan3d tensor summary %q: empty tensor", name)
	}
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
	sampleCount := len(values)
	if sampleCount > 16 {
		sampleCount = 16
	}
	sample := append([]float32(nil), values[:sampleCount]...)
	return TensorSummary{
		Name:        name,
		DType:       "float32",
		Shape:       append([]int(nil), shape...),
		SHA256LEF32: fmt.Sprintf("%x", sha256.Sum256(raw)),
		Min:         minV,
		Max:         maxV,
		Mean:        float32(sum / float64(len(values))),
		FirstValues: sample,
	}, nil
}

func SummarizeImagePreprocessResult(result ImagePreprocessResult) ([]TensorSummary, error) {
	if result.Size <= 0 {
		return nil, fmt.Errorf("hunyuan3d image preprocess summary: invalid size %d", result.Size)
	}
	imageSummary, err := SummarizeFloat32Tensor("image", []int{1, 3, result.Size, result.Size}, result.Image)
	if err != nil {
		return nil, err
	}
	maskSummary, err := SummarizeFloat32Tensor("mask", []int{1, 1, result.Size, result.Size}, result.Mask)
	if err != nil {
		return nil, err
	}
	return []TensorSummary{imageSummary, maskSummary}, nil
}
