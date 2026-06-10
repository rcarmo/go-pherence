package diffusiongemma

import (
	"fmt"

	loaderconfig "github.com/rcarmo/go-pherence/loader/config"
)

// Model is the top-level DiffusionGemma runtime scaffold. It intentionally
// contains metadata and readiness only until tensor loading/forward execution is
// implemented.
type Model struct {
	Path               string                            `json:"path"`
	Config             loaderconfig.DiffusionGemmaConfig `json:"-"`
	Shape              Shape                             `json:"shape"`
	GenerationDefaults *GenerationDefaults               `json:"generation_defaults,omitempty"`
	Denoising          DenoisingConfig                   `json:"denoising"`
	Tensors            *TensorInventory                  `json:"tensors,omitempty"`
	Readiness          *TensorReadiness                  `json:"readiness,omitempty"`
}

func LoadMetadata(modelDir string) (*Model, error) {
	cfg, err := loaderconfig.ReadDiffusionGemmaConfig(modelDir)
	if err != nil {
		return nil, err
	}
	shape, err := ShapeFromConfig(cfg)
	if err != nil {
		return nil, err
	}
	m := &Model{Path: modelDir, Config: cfg, Shape: shape, Denoising: DefaultDenoisingConfig()}
	if gen, ok, err := loaderconfig.ReadDiffusionGemmaGenerationConfig(modelDir); err != nil {
		return nil, err
	} else if ok {
		defaults := GenerationDefaultsFromConfig(gen)
		m.GenerationDefaults = &defaults
		m.Denoising = DenoisingConfigFromDefaults(defaults)
	}
	if inv, ok, err := TensorInventoryFromModelDir(modelDir); err != nil {
		return nil, err
	} else if ok {
		m.Tensors = &inv
		readiness := TensorReadinessFromInventory(inv, shape)
		m.Readiness = &readiness
	}
	return m, nil
}

func (m *Model) RuntimeReady() bool {
	return m != nil && m.Shape.RuntimeReady && m.Readiness != nil && m.Readiness.RuntimeReady
}

func (m *Model) RequireRuntimeReady() error {
	if m == nil {
		return fmt.Errorf("nil DiffusionGemma model")
	}
	if m.RuntimeReady() {
		return nil
	}
	if m.Shape.RuntimeNote != "" {
		return fmt.Errorf("%s", m.Shape.RuntimeNote)
	}
	return fmt.Errorf("DiffusionGemma runtime is not implemented")
}
