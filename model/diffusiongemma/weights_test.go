package diffusiongemma

import (
	"math"
	"os"
	"testing"
)

func TestCachedFloatTensorAppliesFP8Scale(t *testing.T) {
	modelDir := os.Getenv("GO_PHERENCE_DIFFUSIONGEMMA_TEST_MODEL")
	if modelDir == "" {
		t.Skip("set GO_PHERENCE_DIFFUSIONGEMMA_TEST_MODEL to run model-backed FP8 scale test")
	}
	m, err := LoadMetadata(modelDir)
	if err != nil {
		t.Fatal(err)
	}
	weights, err := OpenTextWeights(modelDir, m.Shape)
	if err != nil {
		t.Fatal(err)
	}
	defer weights.Close()
	name := "model.decoder.layers.0.self_attn.q_proj.weight"
	raw, dtype, shape, err := weights.RawTensor(name)
	if err != nil {
		t.Fatal(err)
	}
	if dtype != "F8_E4M3" && dtype != "F8_E4M3FN" {
		t.Fatalf("expected FP8 tensor, got %s", dtype)
	}
	if len(shape) != 2 || shape[0] < 2 || shape[1] < 16 {
		t.Fatalf("unexpected shape %v", shape)
	}
	sRaw, sDType, sShape, err := weights.RawTensor(diffusionGemmaWeightScaleName(name))
	if err != nil {
		t.Fatal(err)
	}
	nScale := 1
	for _, dim := range sShape {
		nScale *= dim
	}
	scales := make([]float32, nScale)
	if err := decodeFloatRowTo(scales, sRaw, sDType); err != nil {
		t.Fatal(err)
	}
	cached, err := weights.CachedFloatTensor(name)
	if err != nil {
		t.Fatal(err)
	}
	for _, row := range []int{0, 1, shape[0] - 1} {
		scale := scales[0]
		if len(scales) != 1 {
			scale = scales[row]
		}
		for col := 0; col < 16; col++ {
			idx := row*shape[1] + col
			want := fp8DecodeE4M3(raw[idx]) * scale
			got := cached.Data[idx]
			if math.Abs(float64(got-want)) > 1e-6 {
				t.Fatalf("cached FP8 scale mismatch row=%d col=%d got=%g want=%g", row, col, got, want)
			}
		}
	}
}
