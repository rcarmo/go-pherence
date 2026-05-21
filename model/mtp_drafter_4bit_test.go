package model

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadGemma4MTPDrafter31B4BitKeepsPackedWeights(t *testing.T) {
	dir := filepath.Join("..", "models", "gemma4-31b-it-mtp-assistant-4bit")
	if _, err := os.Stat(filepath.Join(dir, "model.safetensors")); err != nil {
		t.Skipf("local Gemma4 31B 4-bit MTP assistant asset not available: %v", err)
	}
	d, err := LoadGemma4MTPDrafter(dir)
	if err != nil {
		t.Fatalf("LoadGemma4MTPDrafter: %v", err)
	}
	if d.Config.HiddenSize != 1024 || d.BackboneHiddenSize != 5376 || len(d.Layers) != 4 {
		t.Fatalf("unexpected dims hidden=%d backbone=%d layers=%d", d.Config.HiddenSize, d.BackboneHiddenSize, len(d.Layers))
	}
	if d.EmbedTokensMLX == nil || d.PreProjectionMLX == nil || d.PostProjectionMLX == nil {
		t.Fatalf("packed top-level weights missing embed=%v pre=%v post=%v", d.EmbedTokensMLX != nil, d.PreProjectionMLX != nil, d.PostProjectionMLX != nil)
	}
	if d.EmbedTokens != nil || len(d.PreProjection) != 0 || len(d.PostProjection) != 0 {
		t.Fatalf("4-bit drafter dequantized large weights embed=%v pre=%d post=%d", d.EmbedTokens != nil, len(d.PreProjection), len(d.PostProjection))
	}
	if d.UseOrderedEmbeds || d.MaskedEmbeddingCentroids != nil || d.MaskedEmbeddingOrdering != nil {
		t.Fatalf("31B 4-bit drafter should not require ordered masked embeddings")
	}
	l0 := d.Layers[0]
	if l0.QWm == nil || l0.OWm == nil || l0.GateWm == nil || l0.UpWm == nil || l0.DownWm == nil {
		t.Fatalf("layer 0 packed weights missing q=%v o=%v gate=%v up=%v down=%v", l0.QWm != nil, l0.OWm != nil, l0.GateWm != nil, l0.UpWm != nil, l0.DownWm != nil)
	}
	buf := make([]float32, d.Config.HiddenSize)
	if err := d.AssistantTokenEmbeddingInto(buf, 0); err != nil {
		t.Fatalf("AssistantTokenEmbeddingInto: %v", err)
	}
}
