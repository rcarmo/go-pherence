package lfm2

import (
	"encoding/json"
	"fmt"
	"os"
)

// ReferenceMetadata is a small, commit-safe LFM2 fixture used before runtime
// parity tensors are available. It captures the published config plus optional
// tensor coverage counts from a local safetensors header.
type ReferenceMetadata struct {
	Name    string         `json:"name,omitempty"`
	Config  Config         `json:"config"`
	Tensors TensorCoverage `json:"tensors,omitempty"`
}

func LoadReferenceMetadata(path string) (ReferenceMetadata, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return ReferenceMetadata{}, err
	}
	var meta ReferenceMetadata
	if err := json.Unmarshal(data, &meta); err == nil && meta.Config.ModelType != "" {
		return meta, meta.Validate()
	}
	cfg, err := ParseConfig(data)
	if err != nil {
		return ReferenceMetadata{}, err
	}
	meta = ReferenceMetadata{Name: "config", Config: cfg}
	return meta, nil
}

func (m ReferenceMetadata) Validate() error {
	if err := m.Config.Validate(); err != nil {
		return err
	}
	if m.Tensors.Total < 0 || m.Tensors.Other < 0 {
		return fmt.Errorf("invalid negative tensor coverage: %+v", m.Tensors)
	}
	return nil
}
