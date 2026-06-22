package minicpmv

import (
	"image"
	"image/color"
	"math"
	"testing"
)

func TestNewImageTokenLayout(t *testing.T) {
	layout, err := NewImageTokenLayout(448, 14, 64)
	if err != nil {
		t.Fatalf("NewImageTokenLayout: %v", err)
	}
	if layout.PatchGrid != 32 || layout.VisionTokens != 1024 || layout.ResamplerQuery != 64 {
		t.Fatalf("unexpected layout: %+v", layout)
	}
}

func TestNewImageTokenLayoutRejectsBadPatch(t *testing.T) {
	if _, err := NewImageTokenLayout(450, 14, 64); err == nil {
		t.Fatalf("expected non-divisible image/patch size to fail")
	}
}

func TestPreprocessImageBCHWNormalize(t *testing.T) {
	img := image.NewNRGBA(image.Rect(0, 0, 2, 2))
	img.SetNRGBA(0, 0, color.NRGBA{R: 0, G: 127, B: 255, A: 255})
	img.SetNRGBA(1, 0, color.NRGBA{R: 255, G: 127, B: 0, A: 255})
	img.SetNRGBA(0, 1, color.NRGBA{R: 127, G: 255, B: 0, A: 255})
	img.SetNRGBA(1, 1, color.NRGBA{R: 255, G: 0, B: 127, A: 255})
	cfg := DefaultImagePreprocessConfig(2, 1)
	res, err := PreprocessImage(img, cfg)
	if err != nil {
		t.Fatalf("PreprocessImage: %v", err)
	}
	if res.Shape != [4]int{1, 3, 2, 2} || res.PatchGrid != [2]int{2, 2} || res.PatchCount != 4 {
		t.Fatalf("unexpected result metadata: %+v", res)
	}
	if len(res.PixelValues) != 12 {
		t.Fatalf("pixel len=%d want 12", len(res.PixelValues))
	}
	// BCHW: first plane is red. Default normalize is x/255 then (x-0.5)/0.5.
	wantR := []float32{-1, 1, -0.0039215686, 1}
	for i, want := range wantR {
		if math.Abs(float64(res.PixelValues[i]-want)) > 1e-6 {
			t.Fatalf("red[%d]=%.9f want %.9f", i, res.PixelValues[i], want)
		}
	}
}

func TestPreprocessImageRejectsPatchMismatch(t *testing.T) {
	img := image.NewNRGBA(image.Rect(0, 0, 3, 3))
	cfg := DefaultImagePreprocessConfig(0, 2)
	cfg.DoResize = false
	if _, err := PreprocessImage(img, cfg); err == nil {
		t.Fatalf("expected patch divisibility error")
	}
}

func TestPreprocessImageRejectsZeroStd(t *testing.T) {
	img := image.NewNRGBA(image.Rect(0, 0, 1, 1))
	cfg := DefaultImagePreprocessConfig(1, 1)
	cfg.ImageStd[1] = 0
	if _, err := PreprocessImage(img, cfg); err == nil {
		t.Fatalf("expected zero std error")
	}
}
