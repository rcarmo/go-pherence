package minicpmv

import (
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"testing"
)

func TestPreprocessImageFilePNG(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "img.png")
	img := image.NewNRGBA(image.Rect(0, 0, 2, 2))
	img.SetNRGBA(0, 0, color.NRGBA{R: 255, G: 0, B: 0, A: 255})
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := png.Encode(f, img); err != nil {
		f.Close()
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	res, err := PreprocessImageFile(path, DefaultImagePreprocessConfig(4, 2))
	if err != nil {
		t.Fatalf("PreprocessImageFile: %v", err)
	}
	if res.Path != path || res.Format != "png" || res.Result.Shape != [4]int{1, 3, 4, 4} || res.Result.PatchGrid != [2]int{2, 2} || res.Result.PatchCount != 4 {
		t.Fatalf("bad file preprocess result: %+v", res)
	}
	if len(res.Result.PixelValues) != 3*4*4 {
		t.Fatalf("pixel values len=%d", len(res.Result.PixelValues))
	}
}
