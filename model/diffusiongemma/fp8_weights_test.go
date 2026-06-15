package diffusiongemma

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLocalFP8TextWeightsLoadsSelfConditioningProjections(t *testing.T) {
	modelDir := filepath.Join("..", "..", "models", "diffusiongemma-26B-A4B-it-FP8")
	if _, err := os.Stat(filepath.Join(modelDir, "model.safetensors.index.json")); err != nil {
		t.Skip("local FP8 DiffusionGemma model not present")
	}
	m, err := LoadMetadata(modelDir)
	if err != nil {
		t.Fatal(err)
	}
	w, err := OpenFP8TextWeights(modelDir, m.Shape)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	if len(w.SelfCondPreNorm) != m.Shape.TextHiddenSize {
		t.Fatalf("pre_norm len=%d want %d", len(w.SelfCondPreNorm), m.Shape.TextHiddenSize)
	}
	// DiffusionGemma self-conditioning uses the dense intermediate size, not MoE intermediate.
	wantGate := [2]int{m.Config.TextConfig.IntermediateSize, m.Shape.TextHiddenSize}
	if w.SelfCondGateShape != wantGate || w.SelfCondUpShape != wantGate {
		t.Fatalf("gate/up shapes got %v/%v want %v", w.SelfCondGateShape, w.SelfCondUpShape, wantGate)
	}
	wantDown := [2]int{m.Shape.TextHiddenSize, m.Config.TextConfig.IntermediateSize}
	if w.SelfCondDownShape != wantDown {
		t.Fatalf("down shape got %v want %v", w.SelfCondDownShape, wantDown)
	}
	if len(w.SelfCondGate) != wantGate[0]*wantGate[1] || len(w.SelfCondUp) != wantGate[0]*wantGate[1] || len(w.SelfCondDown) != wantDown[0]*wantDown[1] {
		t.Fatalf("bad self-cond tensor lengths gate=%d up=%d down=%d", len(w.SelfCondGate), len(w.SelfCondUp), len(w.SelfCondDown))
	}
}
