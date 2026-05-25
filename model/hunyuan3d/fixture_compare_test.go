package hunyuan3d

import (
	"os"
	"path/filepath"
	"testing"
)

func TestReadAndCompareImagePreprocessFixture(t *testing.T) {
	summary := TensorSummary{Name: "image", DType: "float32", Shape: []int{1, 3, 2, 2}, SHA256LEF32: "abc", Min: -1, Max: 1, Mean: 0.25}
	mask := TensorSummary{Name: "mask", DType: "float32", Shape: []int{1, 1, 2, 2}, SHA256LEF32: "def", Min: -1, Max: 1, Mean: 0}
	path := filepath.Join(t.TempDir(), "fixture.json")
	data := `{
  "schema": "go-pherence-hunyuan3d-image-preprocess-v1",
  "params": {"size": 2, "border_ratio": 0.15},
  "outputs": [
    {"name":"image","dtype":"float32","shape":[1,3,2,2],"sha256_le_f32":"abc","min":-1,"max":1,"mean":0.25},
    {"name":"mask","dtype":"float32","shape":[1,1,2,2],"sha256_le_f32":"def","min":-1,"max":1,"mean":0}
  ]
}`
	if err := os.WriteFile(path, []byte(data), 0o644); err != nil {
		t.Fatal(err)
	}
	fixture, err := ReadImagePreprocessFixture(path)
	if err != nil {
		t.Fatal(err)
	}
	if fixture.Params.Size != 2 || fixture.Params.BorderRatio != 0.15 {
		t.Fatalf("fixture params=%+v", fixture.Params)
	}
	if err := CompareImagePreprocessSummaries([]TensorSummary{summary, mask}, fixture, 1e-6); err != nil {
		t.Fatal(err)
	}
}

func TestCompareTensorSummaryRejectsMismatch(t *testing.T) {
	base := TensorSummary{Name: "image", DType: "float32", Shape: []int{1}, SHA256LEF32: "a", Min: 0, Max: 0, Mean: 0}
	bad := base
	bad.SHA256LEF32 = "b"
	if err := CompareTensorSummary(base, bad, 1e-6); err == nil {
		t.Fatal("hash mismatch accepted")
	}
	bad = base
	bad.SHA256LEF32 = ""
	bad.Mean = 0.1
	if err := CompareTensorSummary(base, bad, 1e-3); err == nil {
		t.Fatal("numeric mismatch accepted")
	}
	bad = base
	bad.Shape = []int{2}
	if err := CompareTensorSummary(base, bad, 1e-6); err == nil {
		t.Fatal("shape mismatch accepted")
	}
}

func TestReadImagePreprocessFixtureRejectsBadSchema(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bad.json")
	if err := os.WriteFile(path, []byte(`{"schema":"other","params":{"size":1},"outputs":[{}]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := ReadImagePreprocessFixture(path); err == nil {
		t.Fatal("bad schema accepted")
	}
}
