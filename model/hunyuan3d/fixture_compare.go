package hunyuan3d

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"slices"
)

// ImagePreprocessFixture is the JSON schema emitted by
// scripts/hunyuan3d_image_fixture.py. Only stable fields needed for Go parity
// checks are represented here.
type ImagePreprocessFixture struct {
	Schema string `json:"schema"`
	Params struct {
		Size        int     `json:"size"`
		BorderRatio float64 `json:"border_ratio"`
	} `json:"params"`
	Outputs []TensorSummary `json:"outputs"`
}

func ReadImagePreprocessFixture(path string) (ImagePreprocessFixture, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return ImagePreprocessFixture{}, err
	}
	var fixture ImagePreprocessFixture
	if err := json.Unmarshal(data, &fixture); err != nil {
		return ImagePreprocessFixture{}, fmt.Errorf("parse Hunyuan3D image fixture %s: %w", path, err)
	}
	if fixture.Schema != "go-pherence-hunyuan3d-image-preprocess-v1" {
		return ImagePreprocessFixture{}, fmt.Errorf("unsupported Hunyuan3D image fixture schema %q", fixture.Schema)
	}
	if fixture.Params.Size <= 0 {
		return ImagePreprocessFixture{}, fmt.Errorf("invalid Hunyuan3D image fixture size %d", fixture.Params.Size)
	}
	if len(fixture.Outputs) == 0 {
		return ImagePreprocessFixture{}, fmt.Errorf("Hunyuan3D image fixture has no outputs")
	}
	return fixture, nil
}

func FindTensorSummary(outputs []TensorSummary, name string) (TensorSummary, bool) {
	for _, out := range outputs {
		if out.Name == name {
			return out, true
		}
	}
	return TensorSummary{}, false
}

// CompareTensorSummary checks shape/hash equality first and then keeps numeric
// diagnostics available for fixtures that intentionally omit hashes. The hash is
// the authoritative parity signal for float32 fixture tensors.
func CompareTensorSummary(got, want TensorSummary, tolerance float32) error {
	if got.Name != want.Name {
		return fmt.Errorf("tensor summary name mismatch: got %q want %q", got.Name, want.Name)
	}
	if got.DType != want.DType {
		return fmt.Errorf("tensor %s dtype mismatch: got %q want %q", got.Name, got.DType, want.DType)
	}
	if !slices.Equal(got.Shape, want.Shape) {
		return fmt.Errorf("tensor %s shape mismatch: got %v want %v", got.Name, got.Shape, want.Shape)
	}
	if got.SHA256LEF32 != "" && want.SHA256LEF32 != "" && got.SHA256LEF32 != want.SHA256LEF32 {
		return fmt.Errorf("tensor %s sha256_le_f32 mismatch: got %s want %s", got.Name, got.SHA256LEF32, want.SHA256LEF32)
	}
	if err := closeFloat("min", got.Min, want.Min, tolerance); err != nil {
		return fmt.Errorf("tensor %s %w", got.Name, err)
	}
	if err := closeFloat("max", got.Max, want.Max, tolerance); err != nil {
		return fmt.Errorf("tensor %s %w", got.Name, err)
	}
	if err := closeFloat("mean", got.Mean, want.Mean, tolerance); err != nil {
		return fmt.Errorf("tensor %s %w", got.Name, err)
	}
	return nil
}

func CompareImagePreprocessSummaries(got []TensorSummary, fixture ImagePreprocessFixture, tolerance float32) error {
	for _, name := range []string{"image", "mask"} {
		g, ok := FindTensorSummary(got, name)
		if !ok {
			return fmt.Errorf("missing Go tensor summary %q", name)
		}
		w, ok := FindTensorSummary(fixture.Outputs, name)
		if !ok {
			return fmt.Errorf("missing fixture tensor summary %q", name)
		}
		if err := CompareTensorSummary(g, w, tolerance); err != nil {
			return err
		}
	}
	return nil
}

func closeFloat(label string, got, want, tolerance float32) error {
	if float32(math.Abs(float64(got-want))) > tolerance {
		return fmt.Errorf("%s mismatch: got %g want %g tolerance %g", label, got, want, tolerance)
	}
	return nil
}
