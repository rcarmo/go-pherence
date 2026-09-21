package qwen

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/rcarmo/go-pherence/loader/safetensors"
	"github.com/rcarmo/go-pherence/tensor"
)

type Qwen35LoRAConfig struct {
	BaseModel string   `json:"base_model_name_or_path"`
	Revision  string   `json:"revision"`
	Rank      int      `json:"r"`
	Alpha     float32  `json:"lora_alpha"`
	Bias      bool     `json:"lora_bias"`
	TaskType  string   `json:"task_type"`
	Targets   []string `json:"target_modules"`
}

func LoadQwen35LoRA(configPath, weightsPath string) (Qwen35LoRASet, Qwen35LoRAConfig, error) {
	var cfg Qwen35LoRAConfig
	raw, err := os.ReadFile(configPath)
	if err != nil {
		return nil, cfg, err
	}
	if err = json.Unmarshal(raw, &cfg); err != nil {
		return nil, cfg, err
	}
	if cfg.BaseModel == "" || cfg.Revision == "" || cfg.Rank < 1 || cfg.Rank > 4096 || cfg.Alpha <= 0 || cfg.Bias || cfg.TaskType != "CAUSAL_LM" {
		return nil, cfg, fmt.Errorf("qwen: unsupported LoRA config")
	}
	allowed := map[string]bool{}
	for _, target := range cfg.Targets {
		allowed[target] = true
	}
	f, err := safetensors.Open(weightsPath)
	if err != nil {
		return nil, cfg, err
	}
	defer f.Close()
	type pair struct{ a, b *tensor.Tensor }
	pairs := map[string]*pair{}
	const prefix = "base_model.model.model.language_model."
	for _, full := range f.Names() {
		if !strings.HasPrefix(full, prefix) {
			return nil, cfg, fmt.Errorf("qwen: unexpected LoRA tensor %s", full)
		}
		name := strings.TrimPrefix(full, prefix)
		kind := ""
		switch {
		case strings.HasSuffix(name, ".lora_A.weight"):
			name = strings.TrimSuffix(name, ".lora_A.weight")
			kind = "A"
		case strings.HasSuffix(name, ".lora_B.weight"):
			name = strings.TrimSuffix(name, ".lora_B.weight")
			kind = "B"
		default:
			return nil, cfg, fmt.Errorf("qwen: unexpected LoRA tensor %s", full)
		}
		module := name[strings.LastIndex(name, ".")+1:]
		if !allowed[module] {
			return nil, cfg, fmt.Errorf("qwen: undeclared LoRA target %s", name)
		}
		data, shape, err := f.GetFloat32(full)
		if err != nil {
			return nil, cfg, fmt.Errorf("qwen: %s: %w", full, err)
		}
		p := pairs[name]
		if p == nil {
			p = &pair{}
			pairs[name] = p
		}
		t := tensor.FromFloat32(data, shape)
		if kind == "A" {
			if p.a != nil {
				return nil, cfg, fmt.Errorf("qwen: duplicate LoRA A %s", name)
			}
			p.a = t
		} else {
			if p.b != nil {
				return nil, cfg, fmt.Errorf("qwen: duplicate LoRA B %s", name)
			}
			p.b = t
		}
	}
	out := make(Qwen35LoRASet, len(pairs))
	scale := cfg.Alpha / float32(cfg.Rank)
	for name, p := range pairs {
		if p.a == nil || p.b == nil || len(p.a.Shape()) != 2 || len(p.b.Shape()) != 2 || p.a.Shape()[0] != cfg.Rank || p.b.Shape()[1] != cfg.Rank {
			return nil, cfg, fmt.Errorf("qwen: incomplete LoRA target %s", name)
		}
		out[name] = Qwen35LoRA{A: p.a, B: p.b, Scale: scale}
	}
	if len(out) == 0 {
		return nil, cfg, fmt.Errorf("qwen: empty LoRA")
	}
	return out, cfg, nil
}
