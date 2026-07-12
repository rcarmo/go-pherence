package diffusiongemma

import (
	"math"
	"testing"
)

func TestRunVisionTowerF32SyntheticPrefix(t *testing.T) {
	layers := []VisionLayerF32{tinyVisionLayerF32(2, 1, 2, 3), tinyVisionLayerF32(2, 1, 2, 3)}
	hidden := []float32{1, -0.5, 0.25, 0.75}
	if err := RunVisionTowerF32(hidden, 2, 2, 1, 2, layers); err != nil {
		t.Fatal(err)
	}
	want := []float32{5.437500, -3.781250, 0.820312, 6.187500}
	for i := range want {
		if math.Abs(float64(hidden[i]-want[i])) > 1e-5 {
			t.Fatalf("hidden[%d]=%.6f want %.6f full=%v", i, hidden[i], want[i], hidden)
		}
	}
}

func TestRunVisionTowerF32RejectsEmptyLayers(t *testing.T) {
	if err := RunVisionTowerF32([]float32{1, 2}, 1, 2, 1, 2, nil); err == nil {
		t.Fatal("expected empty layers error")
	}
}

func TestLocalDiffusionGemmaRunVisionTowerF32OneLayerSeq1(t *testing.T) {
	dir := localDiffusionGemmaModelDir(t)
	meta, err := LoadMetadata(dir)
	if err != nil {
		t.Fatal(err)
	}
	w, err := OpenVisionWeights(dir, meta.Shape)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	layers, err := LoadVisionTowerF32Prefix(w, 1)
	if err != nil {
		t.Fatal(err)
	}
	hidden := make([]float32, meta.Shape.VisionHiddenSize)
	for i := range hidden {
		hidden[i] = float32((i%7)-3) * 0.01
	}
	headDim := meta.Shape.VisionHiddenSize / meta.Shape.VisionHeads
	if err := RunVisionTowerF32(hidden, 1, meta.Shape.VisionHiddenSize, meta.Shape.VisionHeads, headDim, layers); err != nil {
		t.Fatal(err)
	}
	var sumAbs float64
	for _, v := range hidden {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("non-finite hidden value %v", v)
		}
		sumAbs += math.Abs(float64(v))
	}
	if sumAbs == 0 {
		t.Fatal("vision tower one-layer smoke produced all zeros")
	}
}
