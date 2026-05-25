package hunyuan3d

import (
	"image"
	"image/png"
	"os"
	"testing"
)

// TestImagePreprocessPythonFixtureParity is opt-in because the Python fixture is
// generated locally and may point at a user-supplied image. Enable with:
//
//	GO_PHERENCE_HY3D_IMAGE_FIXTURE=/path/to/fixture.json go test ./model/hunyuan3d
//
// If the fixture source image was file-backed, also set:
//
//	GO_PHERENCE_HY3D_IMAGE=/path/to/source.png
//
// Synthetic fixture parity is intentionally left for a future exact OpenCV
// interpolation pass; this hook validates the comparison path without forcing
// fixture files into git.
func TestImagePreprocessPythonFixtureParity(t *testing.T) {
	fixturePath := os.Getenv("GO_PHERENCE_HY3D_IMAGE_FIXTURE")
	if fixturePath == "" {
		t.Skip("set GO_PHERENCE_HY3D_IMAGE_FIXTURE to validate a local Python image fixture")
	}
	fixture, err := ReadImagePreprocessFixture(fixturePath)
	if err != nil {
		t.Fatal(err)
	}
	imagePath := os.Getenv("GO_PHERENCE_HY3D_IMAGE")
	if imagePath == "" {
		t.Skip("set GO_PHERENCE_HY3D_IMAGE to the fixture source image for Go preprocessing parity")
	}
	src, err := readPNG(imagePath)
	if err != nil {
		t.Fatal(err)
	}
	pre, err := PreprocessImageV2(src, ImagePreprocessConfig{Size: fixture.Params.Size, BorderRatio: fixture.Params.BorderRatio})
	if err != nil {
		t.Fatal(err)
	}
	summaries, err := SummarizeImagePreprocessResult(pre)
	if err != nil {
		t.Fatal(err)
	}
	if err := CompareImagePreprocessSummaries(summaries, fixture, 1e-5); err != nil {
		t.Fatal(err)
	}
}

func readPNG(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return png.Decode(f)
}
