package diffusiongemma

import (
	"image"
	"image/color"
	"math"
	"testing"
)

func TestPreprocessGemma4ImageBCHWRescale(t *testing.T) {
	img := image.NewNRGBA(image.Rect(0, 0, 2, 2))
	img.SetNRGBA(0, 0, color.NRGBA{R: 255, G: 0, B: 0, A: 255})
	img.SetNRGBA(1, 0, color.NRGBA{R: 0, G: 128, B: 0, A: 255})
	img.SetNRGBA(0, 1, color.NRGBA{R: 0, G: 0, B: 64, A: 255})
	img.SetNRGBA(1, 1, color.NRGBA{R: 10, G: 20, B: 30, A: 255})

	cfg := DefaultGemma4ImagePreprocessConfig(0)
	cfg.DoResize = false
	got, err := PreprocessGemma4Image(img, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if got.Shape != [4]int{1, 3, 2, 2} || got.Width != 2 || got.Height != 2 {
		t.Fatalf("shape=%v width=%d height=%d", got.Shape, got.Width, got.Height)
	}
	want := []float32{
		1, 0, 0, 10.0 / 255.0,
		0, 128.0 / 255.0, 0, 20.0 / 255.0,
		0, 0, 64.0 / 255.0, 30.0 / 255.0,
	}
	if len(got.PixelValues) != len(want) {
		t.Fatalf("pixels=%d want %d", len(got.PixelValues), len(want))
	}
	for i := range want {
		if math.Abs(float64(got.PixelValues[i]-want[i])) > 1e-6 {
			t.Fatalf("pixel[%d]=%g want %g all=%v", i, got.PixelValues[i], want[i], got.PixelValues)
		}
	}
}

func TestPreprocessGemma4ImageNormalizeAndResize(t *testing.T) {
	img := image.NewNRGBA(image.Rect(0, 0, 1, 1))
	img.SetNRGBA(0, 0, color.NRGBA{R: 128, G: 64, B: 32, A: 255})
	cfg := Gemma4ImagePreprocessConfig{
		Size:          2,
		DoConvertRGB:  true,
		DoResize:      true,
		DoRescale:     true,
		DoNormalize:   true,
		RescaleFactor: 1.0 / 255.0,
		ImageMean:     [3]float32{0.5, 0.25, 0.125},
		ImageStd:      [3]float32{0.5, 0.25, 0.125},
	}
	got, err := PreprocessGemma4Image(img, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if got.Shape != [4]int{1, 3, 2, 2} {
		t.Fatalf("shape=%v", got.Shape)
	}
	wantFirst := []float32{(128.0/255.0 - 0.5) / 0.5, (64.0/255.0 - 0.25) / 0.25, (32.0/255.0 - 0.125) / 0.125}
	pixels := got.Width * got.Height
	for c, want := range wantFirst {
		if math.Abs(float64(got.PixelValues[c*pixels]-want)) > 1e-6 {
			t.Fatalf("channel %d first=%g want %g", c, got.PixelValues[c*pixels], want)
		}
	}
}

func TestLocalGemma4ImagePreprocessConfigFromProcessor(t *testing.T) {
	dir := localDiffusionGemmaModelDir(t)
	meta, err := LoadMetadata(dir)
	if err != nil {
		t.Fatal(err)
	}
	cfg := Gemma4ImagePreprocessConfigFromProcessor(meta.Processor, 4)
	if !cfg.DoConvertRGB || !cfg.DoResize || !cfg.DoRescale || cfg.DoNormalize {
		t.Fatalf("unexpected processor flags: %+v", cfg)
	}
	if cfg.RescaleFactor != float32(1.0/255.0) || cfg.ImageMean != [3]float32{} || cfg.ImageStd != [3]float32{1, 1, 1} {
		t.Fatalf("unexpected processor numeric config: %+v", cfg)
	}
	img := image.NewNRGBA(image.Rect(0, 0, 2, 2))
	img.SetNRGBA(0, 0, color.NRGBA{R: 255, A: 255})
	got, err := PreprocessGemma4Image(img, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if got.Shape != [4]int{1, 3, 4, 4} {
		t.Fatalf("shape=%v", got.Shape)
	}
}

func TestPreprocessGemma4ImageRejectsBadInputs(t *testing.T) {
	if _, err := PreprocessGemma4Image(nil, DefaultGemma4ImagePreprocessConfig(4)); err == nil {
		t.Fatal("nil image accepted")
	}
	img := image.NewNRGBA(image.Rect(0, 0, 1, 1))
	badResize := DefaultGemma4ImagePreprocessConfig(0)
	badResize.DoResize = true
	if _, err := PreprocessGemma4Image(img, badResize); err == nil {
		t.Fatal("bad resize accepted")
	}
	badStd := DefaultGemma4ImagePreprocessConfig(0)
	badStd.DoResize = false
	badStd.DoNormalize = true
	badStd.ImageStd[1] = 0
	if _, err := PreprocessGemma4Image(img, badStd); err == nil {
		t.Fatal("zero image std accepted")
	}
}
