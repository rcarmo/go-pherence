package omnivoice

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

type tensorReader interface {
	Close() error
	Names() []string
	TensorInfos() map[string]safetensors.TensorInfo
	GetFloat32(string) ([]float32, []int, error)
	GetRaw(string) ([]byte, string, []int, error)
}

// Weights owns a memory mapped single-file or sharded checkpoint. Float32 materializes only
// the requested tensor; the caller can keep at most one decoder layer resident.
// Close after all reads; do not call concurrently with reads.
type Weights struct {
	file   tensorReader
	Config Config
}

func OpenWeights(path string) (*Weights, error) {
	if strings.EqualFold(filepath.Ext(path), ".gguf") {
		return openGGUFWeights(path)
	}
	c, err := LoadConfig(path)
	if err != nil {
		return nil, err
	}
	configPath, err := resolveConfigPath(path)
	if err != nil {
		return nil, err
	}
	dir := filepath.Dir(configPath)
	var f tensorReader
	index := filepath.Join(dir, checkpointIndexName)
	if _, statErr := os.Stat(index); statErr == nil {
		f, err = safetensors.OpenSharded(index)
	} else if os.IsNotExist(statErr) {
		f, err = safetensors.Open(filepath.Join(dir, checkpointFileName))
	} else {
		return nil, statErr
	}
	if err != nil {
		return nil, err
	}
	metadata := ValidateTensorInfos(c, f.TensorInfos())
	if !metadata.Valid {
		f.Close()
		return nil, fmt.Errorf("omnivoice: checkpoint tensor layout mismatch: %+v", metadata)
	}
	return &Weights{file: f, Config: c}, nil
}
func (w *Weights) Close() error { return w.file.Close() }
func (w *Weights) Float32(name string) ([]float32, error) {
	data, _, err := w.file.GetFloat32(name)
	return data, err
}

// Layer materializes one decoder layer, without embedding or audio head tables.
func (w *Weights) Layer(index int) (map[string][]float32, error) {
	if index < 0 || index >= w.Config.LLMConfig.NumHiddenLayers {
		return nil, fmt.Errorf("omnivoice: layer index out of bounds")
	}
	prefix := fmt.Sprintf("llm.layers.%d.", index)
	weights := make(map[string][]float32, 11)
	for _, name := range w.file.Names() {
		if strings.HasPrefix(name, prefix) {
			data, err := w.Float32(name)
			if err != nil {
				return nil, err
			}
			weights[strings.TrimPrefix(name, prefix)] = data
		}
	}
	return weights, nil
}
