package diffusiongemma

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func TestLocalDiffusionGemmaVisionTensorPlanReady(t *testing.T) {
	dir := localDiffusionGemmaModelDir(t)
	meta, err := LoadMetadata(dir)
	if err != nil {
		t.Fatal(err)
	}
	if meta.VisionTensorPlan == nil {
		t.Fatal("vision tensor plan unavailable")
	}
	plan := meta.VisionTensorPlan
	if !plan.Ready || len(plan.Missing) != 0 {
		t.Fatalf("vision plan ready=%v missing=%v", plan.Ready, plan.Missing)
	}
	if len(plan.Globals) != len(diffusionGemmaVisionGlobals) {
		t.Fatalf("globals=%d want %d", len(plan.Globals), len(diffusionGemmaVisionGlobals))
	}
	if len(plan.Layers) != meta.Shape.VisionLayers {
		t.Fatalf("layers=%d want %d", len(plan.Layers), meta.Shape.VisionLayers)
	}
	for _, layer := range plan.Layers {
		if len(layer.Handles) != len(diffusionGemmaVisionLayerSuffixes) {
			t.Fatalf("layer %d handles=%d want %d", layer.Layer, len(layer.Handles), len(diffusionGemmaVisionLayerSuffixes))
		}
		for _, h := range layer.Handles {
			if h.Shard == "" || !h.Required {
				t.Fatalf("bad handle layer=%d handle=%+v", layer.Layer, h)
			}
		}
	}
}

func TestLocalDiffusionGemmaVisionTensorPlanShapes(t *testing.T) {
	dir := localDiffusionGemmaModelDir(t)
	meta, err := LoadMetadata(dir)
	if err != nil {
		t.Fatal(err)
	}
	sf, err := safetensors.OpenSharded(filepath.Join(dir, "model.safetensors.index.json"))
	if err != nil {
		t.Fatal(err)
	}
	defer sf.Close()
	mismatches := ValidateVisionTensorPlanShapes(*meta.VisionTensorPlan, sf.TensorInfos(), meta.Shape)
	if len(mismatches) != 0 {
		t.Fatalf("vision shape mismatches: %v", mismatches)
	}
}

func TestVisionTensorPlanFromIndexReportsMissing(t *testing.T) {
	idx := writeVisionTensorPlanIndex(t, map[string]string{
		"model.encoder.vision_tower.patch_embedder.input_proj.weight": "model.safetensors",
	})
	plan, err := VisionTensorPlanFromIndex(idx, Shape{VisionLayers: 1})
	if err != nil {
		t.Fatal(err)
	}
	if plan.Ready || len(plan.Missing) == 0 {
		t.Fatalf("expected missing vision tensors, got ready=%v missing=%v", plan.Ready, plan.Missing)
	}
}

func writeVisionTensorPlanIndex(t *testing.T, weightMap map[string]string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "model.safetensors.index.json")
	body := struct {
		Metadata  map[string]int64  `json:"metadata"`
		WeightMap map[string]string `json:"weight_map"`
	}{Metadata: map[string]int64{}, WeightMap: weightMap}
	b, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}
