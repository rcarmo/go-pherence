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
	TextTensorPlan     *TextTensorPlan                   `json:"text_tensor_plan,omitempty"`
	Processor          *ProcessorMetadata                `json:"processor,omitempty"`
	Tokenizer          *TokenizerMetadata                `json:"tokenizer,omitempty"`
	Shards             *ShardAvailability                `json:"shards,omitempty"`
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
	if plan, ok, err := TextTensorPlanFromModelDir(modelDir, shape); err != nil {
		return nil, err
	} else if ok {
		m.TextTensorPlan = &plan
	}
	if shards, ok, err := ShardAvailabilityFromModelDir(modelDir); err != nil {
		return nil, err
	} else if ok {
		m.Shards = &shards
	}
	if proc, ok, err := ReadProcessorMetadata(modelDir); err != nil {
		return nil, err
	} else if ok {
		m.Processor = &proc
		tokens := []string{proc.BOS, proc.EOS, proc.Pad, proc.Mask, proc.BOI, proc.EOI, proc.Image, proc.Think, proc.BOT, proc.EOT, proc.BOC, proc.EOC}
		if tok, tokOK, err := ReadTokenizerMetadata(modelDir, compactStrings(tokens)); err != nil {
			return nil, err
		} else if tokOK {
			m.Tokenizer = &tok
		}
	}
	return m, nil
}

func compactStrings(in []string) []string {
	seen := map[string]bool{}
	out := make([]string, 0, len(in))
	for _, s := range in {
		if s == "" || seen[s] {
			continue
		}
		seen[s] = true
		out = append(out, s)
	}
	return out
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
