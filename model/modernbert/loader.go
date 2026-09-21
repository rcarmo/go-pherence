package modernbert

import (
	"encoding/json"
	"fmt"
	"github.com/rcarmo/go-pherence/loader/safetensors"
	"os"
	"path/filepath"
	"strings"
)

func Load(modelDir string) (*Model, error) {
	return LoadFiles(filepath.Join(modelDir, "config.json"), filepath.Join(modelDir, "model.safetensors"), "model.")
}

// LoadFiles loads ModernBERT tensors from a strict prefix, allowing composite
// checkpoints (for example Laya's encoder.*) without accepting unrelated heads.
func LoadFiles(configPath, tensorPath, prefix string) (*Model, error) {
	if prefix == "" {
		return nil, fmt.Errorf("modernbert: empty tensor prefix")
	}
	b, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("modernbert: config: %w", err)
	}
	f, err := safetensors.Open(tensorPath)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	tensors := map[string]struct {
		Shape []int
		Data  []float32
	}{}
	for _, name := range f.Names() {
		if !strings.HasPrefix(name, prefix) {
			continue
		}
		short := strings.TrimPrefix(name, prefix)
		data, shape, err := f.GetFloat32(name)
		if err != nil {
			return nil, fmt.Errorf("modernbert: %s: %w", name, err)
		}
		tensors[short] = struct {
			Shape []int
			Data  []float32
		}{shape, data}
	}
	// Convert through the checkpoint tensor type at the package boundary without
	// retaining mmap-backed storage.
	cp := make(map[string]Tensor, len(tensors))
	for k, v := range tensors {
		cp[k] = Tensor{Shape: v.Shape, Data: v.Data}
	}
	return newModel(json.RawMessage(b), cp, true)
}
