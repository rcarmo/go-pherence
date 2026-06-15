package diffusiongemma

import (
	"math"
	"testing"
)

func TestRunVisionTowerF32StreamingEquivalentSynthetic(t *testing.T) {
	// Synthetic equivalent of streaming: run one layer at a time and compare with
	// the pre-materialized tower helper.
	layers := []VisionLayerF32{tinyVisionLayerF32(2, 1, 2, 3), tinyVisionLayerF32(2, 1, 2, 3)}
	batch := []float32{1, -0.5, 0.25, 0.75}
	stream := append([]float32(nil), batch...)
	if err := RunVisionTowerF32(batch, 2, 2, 1, 2, layers); err != nil {
		t.Fatal(err)
	}
	for i, layer := range layers {
		if err := RunVisionLayerF32(stream, 2, 2, 1, 2, layer); err != nil {
			t.Fatalf("stream layer %d: %v", i, err)
		}
	}
	for i := range batch {
		if math.Abs(float64(batch[i]-stream[i])) > 1e-6 {
			t.Fatalf("value[%d] batch=%.7f stream=%.7f batch_all=%v stream_all=%v", i, batch[i], stream[i], batch, stream)
		}
	}
}

func TestLocalDiffusionGemmaRunVisionTowerF32StreamingPrefixSeq1(t *testing.T) {
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
	hidden := make([]float32, meta.Shape.VisionHiddenSize)
	for i := range hidden {
		hidden[i] = float32((i%11)-5) * 0.005
	}
	if err := RunVisionTowerF32StreamingPrefix(hidden, 1, meta.Shape, w, 2); err != nil {
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
		t.Fatal("streaming vision tower prefix produced all zeros")
	}
}

func TestLocalDiffusionGemmaRunVisionTowerF32StreamingFullDepthSeq1(t *testing.T) {
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
	hidden := make([]float32, meta.Shape.VisionHiddenSize)
	for i := range hidden {
		hidden[i] = float32((i%23)-11) * 0.001
	}
	if err := RunVisionTowerF32StreamingPrefix(hidden, 1, meta.Shape, w, meta.Shape.VisionLayers); err != nil {
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
		t.Fatal("full-depth streaming vision tower produced all zeros")
	}
}

func TestRunVisionTowerF32StreamingPrefixRejectsCount(t *testing.T) {
	if err := RunVisionTowerF32StreamingPrefix([]float32{1, 2}, 1, Shape{VisionHiddenSize: 2, VisionHeads: 1}, nil, 1); err == nil {
		t.Fatal("expected nil weights error")
	}
}
