package model

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLoadGemma4MTPDrafter31B4BitDocumentsCurrentBlocker(t *testing.T) {
	dir := filepath.Join("..", "models", "gemma4-31b-it-mtp-assistant-4bit")
	if _, err := os.Stat(filepath.Join(dir, "model.safetensors")); err != nil {
		t.Skipf("local Gemma4 31B 4-bit MTP assistant asset not available: %v", err)
	}
	_, err := LoadGemma4MTPDrafter(dir)
	if err == nil {
		t.Fatal("LoadGemma4MTPDrafter unexpectedly accepted 4-bit MLX assistant weights")
	}
	if !strings.Contains(err.Error(), "unsupported dtype \"U32\"") {
		t.Fatalf("LoadGemma4MTPDrafter err=%v, want U32/MLX quantization blocker", err)
	}
}
