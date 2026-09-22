package pockettts

import (
	"fmt"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func LoadGeneratorCPU(path string, cfg Config) (*GeneratorCPU, error) {
	file, err := safetensors.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	lm, err := LoadFlowLMCPU(file, cfg)
	if err != nil {
		return nil, err
	}
	flow, err := LoadFlowHeadCPU(file, cfg)
	if err != nil {
		return nil, err
	}
	mimi, err := LoadMimiDecoderCPU(file, cfg)
	if err != nil {
		return nil, err
	}
	mean, err := loadVectorF32(file, "flow_lm.emb_mean", cfg.Mimi.InnerDim)
	if err != nil {
		return nil, err
	}
	std, err := loadVectorF32(file, "flow_lm.emb_std", cfg.Mimi.InnerDim)
	if err != nil {
		return nil, err
	}
	for i, value := range std {
		if value <= 0 {
			return nil, fmt.Errorf("Pocket TTS latent std[%d]=%g", i, value)
		}
	}
	return NewGeneratorCPU(lm, flow, mimi, mean, std)
}
