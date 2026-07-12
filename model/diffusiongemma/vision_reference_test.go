package diffusiongemma

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

type gemma4Vision48Reference struct {
	Schema      int       `json:"schema"`
	PatchFirst  []float32 `json:"patch_first"`
	Layer1First []float32 `json:"layer1_first"`
	Final       []float32 `json:"final"`
}

func TestLocalDiffusionGemmaVision48TransformersReference(t *testing.T) {
	dir := localDiffusionGemmaModelDir(t)
	data, err := os.ReadFile("testdata/gemma4_vision_48x48_transformers.json")
	if err != nil {
		t.Fatal(err)
	}
	var ref gemma4Vision48Reference
	if err := json.Unmarshal(data, &ref); err != nil {
		t.Fatal(err)
	}
	if ref.Schema != 1 {
		t.Fatalf("fixture schema=%d want 1", ref.Schema)
	}
	meta, err := LoadMetadata(dir)
	if err != nil {
		t.Fatal(err)
	}
	w, err := OpenVisionWeights(dir, meta.Shape)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	const width, height = 48, 48
	pixels := make([]float32, 3*width*height)
	for c := 0; c < 3; c++ {
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				pixels[c*width*height+y*width+x] = float32((x*17+y*13+c*71)%256) / 255
			}
		}
	}
	shape := meta.Shape
	shape.PatchSize = 16
	shape.VisionSoftTokens = 1
	pre := Gemma4ImagePreprocessResult{PixelValues: pixels, Shape: [4]int{1, 3, height, width}, Width: width, Height: height}
	patchHidden, _, err := computeImagePatchHidden(pre, w, shape)
	if err != nil {
		t.Fatal(err)
	}
	assertVisionReferenceMetrics(t, "patch", patchHidden[:shape.VisionHiddenSize], ref.PatchFirst)
	layer1 := append([]float32(nil), patchHidden...)
	if err := ApplyVisionTowerStreamingPrefixF32(layer1, 9, shape, w, 1, 3, 3); err != nil {
		t.Fatal(err)
	}
	assertVisionReferenceMetrics(t, "layer1", layer1[:shape.VisionHiddenSize], ref.Layer1First)
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_VISION_FULL_STREAMING_MAX_PATCHES", "9")
	got, err := ComputeImageEmbeddingsWithFullStreamingTower(pre, w, shape)
	if err != nil {
		t.Fatal(err)
	}
	assertVisionReferenceMetrics(t, "final", got.Embeddings, ref.Final)
}

func assertVisionReferenceMetrics(t *testing.T, name string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s len=%d want %d", name, len(got), len(want))
	}
	var dot, gg, ww, absSum, maxAbs float64
	for i := range got {
		g, w := float64(got[i]), float64(want[i])
		d := math.Abs(g - w)
		dot += g * w
		gg += g * g
		ww += w * w
		absSum += d
		if d > maxAbs {
			maxAbs = d
		}
	}
	cosine := dot / math.Sqrt(gg*ww)
	meanAbs := absSum / float64(len(got))
	t.Logf("%s cosine=%.9f mean_abs=%.6f max_abs=%.6f", name, cosine, meanAbs, maxAbs)
	// Initial broad guards are tightened as backend accumulation is aligned.
	if cosine < 0.98 || meanAbs > 0.25 || maxAbs > 2.0 {
		t.Fatalf("%s reference drift cosine=%.9f mean_abs=%.6f max_abs=%.6f", name, cosine, meanAbs, maxAbs)
	}
}
