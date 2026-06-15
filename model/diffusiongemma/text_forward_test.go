package diffusiongemma

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestAssignLayerBindingDistinguishesEncoderLayerScalar(t *testing.T) {
	var lb TextLayerBindings
	dec := TensorBinding{TensorHandle: TensorHandle{Name: "model.decoder.layers.0.layer_scalar"}}
	enc := TensorBinding{TensorHandle: TensorHandle{Name: "model.encoder.language_model.layers.0.layer_scalar"}}

	assignLayerBinding(&lb, &dec)
	assignLayerBinding(&lb, &enc)

	if lb.LayerScalar == nil || lb.LayerScalar.Name != dec.Name {
		t.Fatalf("decoder layer scalar not bound correctly: %+v", lb.LayerScalar)
	}
	if lb.EncLayerScalar == nil || lb.EncLayerScalar.Name != enc.Name {
		t.Fatalf("encoder layer scalar not bound correctly: %+v", lb.EncLayerScalar)
	}
}

func TestTextTensorPlanIncludesOptionalEncoderLayerScalar(t *testing.T) {
	weightMap := map[string]string{}
	add := func(name string) { weightMap[name] = "model.safetensors" }
	for _, name := range diffusionGemmaRequiredGlobals {
		add(name)
	}
	base := "model.decoder.layers.0."
	for _, suffix := range requiredLayerSuffixesForType("sliding_attention") {
		add(base + suffix)
	}
	encScalar := "model.encoder.language_model.layers.0.layer_scalar"
	add(encScalar)

	idx := hfSafetensorsIndex{WeightMap: weightMap}
	data, err := json.Marshal(idx)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "model.safetensors.index.json")
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}

	plan, err := TextTensorPlanFromIndex(path, Shape{TextLayers: 1, LayerTypes: []string{"sliding_attention"}})
	if err != nil {
		t.Fatal(err)
	}
	if !plan.Ready {
		t.Fatalf("plan unexpectedly not ready: missing=%v", plan.Missing)
	}
	found := false
	for _, h := range plan.Layers[0].Handles {
		if h.Name == encScalar {
			found = true
			if h.Required {
				t.Fatalf("encoder layer scalar should be optional")
			}
		}
	}
	if !found {
		t.Fatalf("encoder layer scalar handle not present in plan")
	}
}
