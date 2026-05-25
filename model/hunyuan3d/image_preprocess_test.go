package hunyuan3d

import (
	"image"
	"image/color"
	"testing"
)

func TestPreprocessImageV2Synthetic(t *testing.T) {
	src := image.NewNRGBA(image.Rect(0, 0, 12, 8))
	for y := 2; y < 6; y++ {
		for x := 2; x < 5; x++ {
			src.SetNRGBA(x, y, color.NRGBA{R: uint8(30 + x), G: uint8(80 + y), B: 160, A: 255})
		}
	}
	got, err := PreprocessImageV2(src, ImagePreprocessConfig{Size: 16, BorderRatio: 0.25})
	if err != nil {
		t.Fatal(err)
	}
	if got.Size != 16 || len(got.Image) != 3*16*16 || len(got.Mask) != 16*16 {
		t.Fatalf("unexpected output shape size=%d image=%d mask=%d", got.Size, len(got.Image), len(got.Mask))
	}
	var maskActive int
	for _, v := range got.Mask {
		if v > -1 {
			maskActive++
		}
	}
	if maskActive == 0 {
		t.Fatal("mask has no active pixels")
	}
	center := 8*16 + 8
	if got.Mask[center] <= -1 {
		t.Fatalf("object was not recentered near output center: center mask=%v", got.Mask[center])
	}
	for i, v := range got.Image {
		if v < -1 || v > 1 {
			t.Fatalf("image[%d]=%v out of range", i, v)
		}
	}
}

func TestPreprocessImageV2RejectsEmptyAndBadConfig(t *testing.T) {
	if _, err := PreprocessImageV2(nil, DefaultImagePreprocessConfig()); err == nil {
		t.Fatal("nil image accepted")
	}
	img := image.NewNRGBA(image.Rect(0, 0, 4, 4))
	if _, err := PreprocessImageV2(img, DefaultImagePreprocessConfig()); err == nil {
		t.Fatal("empty alpha mask accepted")
	}
	img.SetNRGBA(1, 1, color.NRGBA{A: 255})
	if _, err := PreprocessImageV2(img, ImagePreprocessConfig{Size: 0, BorderRatio: 0.1}); err == nil {
		t.Fatal("bad size accepted")
	}
	if _, err := PreprocessImageV2(img, ImagePreprocessConfig{Size: 8, BorderRatio: 1}); err == nil {
		t.Fatal("bad border ratio accepted")
	}
}
