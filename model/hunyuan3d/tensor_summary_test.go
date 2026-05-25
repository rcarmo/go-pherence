package hunyuan3d

import (
	"image"
	"image/color"
	"testing"
)

func TestSummarizeFloat32Tensor(t *testing.T) {
	got, err := SummarizeFloat32Tensor("x", []int{2, 2}, []float32{1, -1, 3, 5})
	if err != nil {
		t.Fatal(err)
	}
	if got.Name != "x" || got.DType != "float32" || got.Min != -1 || got.Max != 5 || got.Mean != 2 {
		t.Fatalf("summary=%+v", got)
	}
	if got.SHA256LEF32 != "90f73a7925d83a9ec35185cd7814055c7ad4e82c6cd791d53744d130781ab610" {
		t.Fatalf("hash=%s", got.SHA256LEF32)
	}
	if len(got.FirstValues) != 4 || got.FirstValues[2] != 3 {
		t.Fatalf("first_values=%v", got.FirstValues)
	}
	got.FirstValues[0] = 99
	if got.Shape[0] != 2 {
		t.Fatalf("shape mutated unexpectedly: %v", got.Shape)
	}
	if _, err := SummarizeFloat32Tensor("bad", []int{3}, []float32{1, 2}); err == nil {
		t.Fatal("shape mismatch accepted")
	}
}

func TestSummarizeImagePreprocessResult(t *testing.T) {
	img := image.NewNRGBA(image.Rect(0, 0, 8, 8))
	for y := 2; y < 6; y++ {
		for x := 3; x < 6; x++ {
			img.SetNRGBA(x, y, color.NRGBA{R: 120, G: 80, B: 40, A: 255})
		}
	}
	pre, err := PreprocessImageV2(img, ImagePreprocessConfig{Size: 8, BorderRatio: 0.25})
	if err != nil {
		t.Fatal(err)
	}
	summaries, err := SummarizeImagePreprocessResult(pre)
	if err != nil {
		t.Fatal(err)
	}
	if len(summaries) != 2 {
		t.Fatalf("summaries=%d", len(summaries))
	}
	if summaries[0].Name != "image" || len(summaries[0].Shape) != 4 || summaries[0].Shape[1] != 3 {
		t.Fatalf("image summary=%+v", summaries[0])
	}
	if summaries[1].Name != "mask" || summaries[1].Shape[1] != 1 {
		t.Fatalf("mask summary=%+v", summaries[1])
	}
	if summaries[0].SHA256LEF32 == "" || summaries[1].SHA256LEF32 == "" {
		t.Fatalf("missing hashes: %+v", summaries)
	}
}
