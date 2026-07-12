package diffusiongemma

import (
	"math"
	"strings"
	"testing"
)

func TestApplyVisionTowerPrefixF32NilNoop(t *testing.T) {
	vision := []float32{1, 2, 3, 4}
	orig := append([]float32(nil), vision...)
	shape := Shape{VisionHiddenSize: 2, VisionHeads: 1}
	if err := ApplyVisionTowerPrefixF32(vision, 2, shape, nil); err != nil {
		t.Fatal(err)
	}
	for i := range orig {
		if vision[i] != orig[i] {
			t.Fatalf("vision[%d]=%v want %v", i, vision[i], orig[i])
		}
	}
}

func TestApplyVisionTowerPrefixF32Synthetic(t *testing.T) {
	vision := []float32{1, -0.5, 0.25, 0.75}
	shape := Shape{VisionHiddenSize: 2, VisionHeads: 1}
	if err := ApplyVisionTowerPrefixF32(vision, 2, shape, []VisionLayerF32{tinyVisionLayerF32(2, 1, 2, 3)}); err != nil {
		t.Fatal(err)
	}
	want := []float32{3.234375, -2.062500, 0.388672, 3.375000}
	for i := range want {
		if math.Abs(float64(vision[i]-want[i])) > 1e-5 {
			t.Fatalf("vision[%d]=%.6f want %.6f full=%v", i, vision[i], want[i], vision)
		}
	}
}

func TestApplyVisionTowerPrefixF32RejectsBadShape(t *testing.T) {
	vision := []float32{1, 2, 3, 4}
	shape := Shape{VisionHiddenSize: 3, VisionHeads: 1}
	if err := ApplyVisionTowerPrefixF32(vision, 2, shape, []VisionLayerF32{tinyVisionLayerF32(2, 1, 2, 3)}); err == nil {
		t.Fatal("expected invalid hidden shape error")
	}
}

func TestComputeImageEmbeddingsWithFullStreamingTowerRequiresLayers(t *testing.T) {
	_, err := ComputeImageEmbeddingsWithFullStreamingTower(Gemma4ImagePreprocessResult{}, nil, Shape{})
	if err == nil {
		t.Fatal("expected missing vision layers error")
	}
}

func TestLocalDiffusionGemmaStreamingImageEmbeddingsZeroPrefixMatchesNoTower(t *testing.T) {
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
	shape := meta.Shape
	shape.PatchSize = 16
	shape.VisionSoftTokens = 1
	pre := Gemma4ImagePreprocessResult{
		PixelValues: make([]float32, 3*16*16),
		Shape:       [4]int{1, 3, 16, 16},
		Width:       16,
		Height:      16,
	}
	for i := range pre.PixelValues {
		pre.PixelValues[i] = float32((i%17)-8) * 0.01
	}
	base, err := ComputeImageEmbeddings(pre, w, shape)
	if err != nil {
		t.Fatal(err)
	}
	streaming, err := ComputeImageEmbeddingsWithStreamingTowerPrefix(pre, w, shape, 0)
	if err != nil {
		t.Fatal(err)
	}
	if base.Shape != streaming.Shape || len(base.Embeddings) != len(streaming.Embeddings) {
		t.Fatalf("shape mismatch base=%v/%d streaming=%v/%d", base.Shape, len(base.Embeddings), streaming.Shape, len(streaming.Embeddings))
	}
	for i := range base.Embeddings {
		if math.Abs(float64(base.Embeddings[i]-streaming.Embeddings[i])) > 1e-6 {
			t.Fatalf("embedding[%d] base=%.7f streaming=%.7f", i, base.Embeddings[i], streaming.Embeddings[i])
		}
	}
}

func TestLocalDiffusionGemmaStreamingImageEmbeddingsOneLayerMatchesPreloaded(t *testing.T) {
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
	shape := meta.Shape
	shape.PatchSize = 16
	shape.VisionSoftTokens = 1
	preloaded, err := LoadVisionTowerF32Prefix(w, 1)
	if err != nil {
		t.Fatal(err)
	}
	pre := Gemma4ImagePreprocessResult{
		PixelValues: make([]float32, 3*16*16),
		Shape:       [4]int{1, 3, 16, 16},
		Width:       16,
		Height:      16,
	}
	for i := range pre.PixelValues {
		pre.PixelValues[i] = float32((i%19)-9) * 0.0075
	}
	want, err := ComputeImageEmbeddingsWithTowerPrefix(pre, w, shape, preloaded)
	if err != nil {
		t.Fatal(err)
	}
	got, err := ComputeImageEmbeddingsWithStreamingTowerPrefix(pre, w, shape, 1)
	if err != nil {
		t.Fatal(err)
	}
	if want.Shape != got.Shape || len(want.Embeddings) != len(got.Embeddings) {
		t.Fatalf("shape mismatch preloaded=%v/%d streaming=%v/%d", want.Shape, len(want.Embeddings), got.Shape, len(got.Embeddings))
	}
	for i := range want.Embeddings {
		if math.Abs(float64(want.Embeddings[i]-got.Embeddings[i])) > 1e-6 {
			t.Fatalf("embedding[%d] preloaded=%.7f streaming=%.7f", i, want.Embeddings[i], got.Embeddings[i])
		}
	}
}

func TestComputeImageEmbeddingsWithFullStreamingTowerGuardRejectsLargePatchCount(t *testing.T) {
	pre := Gemma4ImagePreprocessResult{Shape: [4]int{1, 3, 32, 32}, Width: 32, Height: 32}
	shape := Shape{PatchSize: 16, VisionLayers: 1}
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_VISION_FULL_STREAMING_MAX_PATCHES", "1")
	_, err := ComputeImageEmbeddingsWithFullStreamingTower(pre, nil, shape)
	if err == nil {
		t.Fatal("expected full streaming patch-count guard error")
	}
}

func TestComputeImageEmbeddingsWithFullStreamingTowerGuardAllowsOverride(t *testing.T) {
	pre := Gemma4ImagePreprocessResult{Shape: [4]int{1, 3, 32, 32}, Width: 32, Height: 32}
	shape := Shape{PatchSize: 16, VisionLayers: 1}
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_VISION_FULL_STREAMING_MAX_PATCHES", "4")
	_, err := ComputeImageEmbeddingsWithFullStreamingTower(pre, nil, shape)
	if err == nil || !strings.Contains(err.Error(), "missing vision weights") {
		t.Fatalf("expected guard override to reach missing-weights error, got %v", err)
	}
}

func TestLocalDiffusionGemmaFullStreamingTowerGuardRejectsProcessorImageSeq(t *testing.T) {
	dir := localDiffusionGemmaModelDir(t)
	meta, err := LoadMetadata(dir)
	if err != nil {
		t.Fatal(err)
	}
	if meta.Processor == nil || meta.Processor.ImageSeqLength <= 0 {
		t.Fatal("local processor image sequence metadata missing")
	}
	shape := meta.Shape
	patchW := 20
	if meta.Processor.ImageSeqLength%patchW != 0 {
		t.Fatalf("local processor image_seq=%d not divisible by test patch width %d", meta.Processor.ImageSeqLength, patchW)
	}
	patchH := meta.Processor.ImageSeqLength / patchW
	width, height := patchW*shape.PatchSize, patchH*shape.PatchSize
	pre := Gemma4ImagePreprocessResult{Shape: [4]int{1, 3, height, width}, Width: width, Height: height}
	patches, err := imagePatchCount(pre, shape)
	if err != nil {
		t.Fatal(err)
	}
	if patches != meta.Processor.ImageSeqLength {
		t.Fatalf("constructed patches=%d want processor image_seq=%d", patches, meta.Processor.ImageSeqLength)
	}
	_, err = ComputeImageEmbeddingsWithFullStreamingTower(pre, nil, shape)
	if err == nil || !strings.Contains(err.Error(), "exceeds guarded CPU scaffold limit") {
		t.Fatalf("expected full image-sequence guard error, got %v", err)
	}
}

func TestLocalDiffusionGemmaFullStreamingImageEmbeddingsOnePatch(t *testing.T) {
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
	shape := meta.Shape
	shape.PatchSize = 16
	shape.VisionSoftTokens = 1
	pre := Gemma4ImagePreprocessResult{
		PixelValues: make([]float32, 3*16*16),
		Shape:       [4]int{1, 3, 16, 16},
		Width:       16,
		Height:      16,
	}
	for i := range pre.PixelValues {
		pre.PixelValues[i] = float32((i%29)-14) * 0.003
	}
	got, err := ComputeImageEmbeddingsWithFullStreamingTower(pre, w, shape)
	if err != nil {
		t.Fatal(err)
	}
	if got.Shape != [3]int{1, 1, meta.Shape.TextHiddenSize} || len(got.Embeddings) != meta.Shape.TextHiddenSize {
		t.Fatalf("embedding shape=%v len=%d want [1 1 %d]", got.Shape, len(got.Embeddings), meta.Shape.TextHiddenSize)
	}
	var sumAbs float64
	for _, v := range got.Embeddings {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("non-finite embedding value %v", v)
		}
		sumAbs += math.Abs(float64(v))
	}
	if sumAbs == 0 {
		t.Fatal("full streaming image embedding produced all zeros")
	}
}

func TestApplyVisionTowerStreamingPrefixF32CountZeroNoop(t *testing.T) {
	vision := []float32{1, 2, 3, 4}
	orig := append([]float32(nil), vision...)
	shape := Shape{VisionHiddenSize: 2, VisionHeads: 1}
	if err := ApplyVisionTowerStreamingPrefixF32(vision, 2, shape, nil, 0); err != nil {
		t.Fatal(err)
	}
	for i := range orig {
		if vision[i] != orig[i] {
			t.Fatalf("vision[%d]=%v want %v", i, vision[i], orig[i])
		}
	}
}

func TestLocalDiffusionGemmaStreamingTowerHookMatchesPreloadedPrefixSeq1(t *testing.T) {
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
	preloaded, err := LoadVisionTowerF32Prefix(w, 1)
	if err != nil {
		t.Fatal(err)
	}
	base := make([]float32, meta.Shape.VisionHiddenSize)
	for i := range base {
		base[i] = float32((i%13)-6) * 0.0025
	}
	want := append([]float32(nil), base...)
	got := append([]float32(nil), base...)
	if err := ApplyVisionTowerPrefixF32(want, 1, meta.Shape, preloaded); err != nil {
		t.Fatal(err)
	}
	if err := ApplyVisionTowerStreamingPrefixF32(got, 1, meta.Shape, w, 1); err != nil {
		t.Fatal(err)
	}
	for i := range want {
		if math.Abs(float64(want[i]-got[i])) > 1e-6 {
			t.Fatalf("hidden[%d] preloaded=%.7f streaming=%.7f", i, want[i], got[i])
		}
	}
}
