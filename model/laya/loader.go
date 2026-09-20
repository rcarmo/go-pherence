package laya

import (
	"encoding/json"
	"fmt"
	"github.com/rcarmo/go-pherence/loader/safetensors"
	"github.com/rcarmo/go-pherence/model/modernbert"
	"os"
	"path/filepath"
	"strings"
)

type Config struct {
	MaxLen               int                `json:"max_len"`
	HeadMaxLen           int                `json:"head_max_len"`
	HeadLayers           int                `json:"head_layers"`
	Encoder              string             `json:"encoder"`
	ActCosts             map[string]float64 `json:"act_costs"`
	Temperature          []float32          `json:"temperature"`
	TemperatureByOptions map[string]float32 `json:"temperature_by_options"`
}

func Load(modelDir, encoderConfig string) (*Model, Config, error) {
	var cfg Config
	b, e := os.ReadFile(filepath.Join(modelDir, "rl_agent_config.json"))
	if e != nil {
		return nil, cfg, e
	}
	if e = json.Unmarshal(b, &cfg); e != nil {
		return nil, cfg, e
	}
	if cfg.MaxLen < 1 || cfg.MaxLen > 65536 || cfg.HeadMaxLen < 16 || cfg.HeadMaxLen > cfg.MaxLen || cfg.HeadLayers < 0 || cfg.HeadLayers > 32 {
		return nil, cfg, fmt.Errorf("laya: invalid config")
	}
	enc, e := modernbert.LoadFiles(encoderConfig, filepath.Join(modelDir, "model.safetensors"), "encoder.")
	if e != nil {
		return nil, cfg, e
	}
	f, e := safetensors.Open(filepath.Join(modelDir, "model.safetensors"))
	if e != nil {
		return nil, cfg, e
	}
	defer f.Close()
	tensors := map[string]Tensor{}
	for _, name := range f.Names() {
		if strings.HasPrefix(name, "encoder.") {
			continue
		}
		data, shape, e := f.GetFloat32(name)
		if e != nil {
			return nil, cfg, fmt.Errorf("laya: %s: %w", name, e)
		}
		tensors[name] = Tensor{Shape: shape, Data: data}
	}
	m, e := newModel(enc, tensors, false)
	if e != nil {
		return nil, cfg, e
	}
	if len(cfg.Temperature) != 0 {
		if len(cfg.Temperature) != 3 {
			return nil, cfg, fmt.Errorf("laya: invalid temperature")
		}
		for i, value := range cfg.Temperature {
			if value < 1e-3 {
				return nil, cfg, fmt.Errorf("laya: invalid temperature")
			}
			m.temp[i] = value
		}
	}
	m.tempByOptions = make(map[string]float32, len(cfg.TemperatureByOptions))
	for bucket, value := range cfg.TemperatureByOptions {
		if value < 1e-3 || !validTemperatureBucket(bucket) {
			return nil, cfg, fmt.Errorf("laya: invalid temperature bucket %q", bucket)
		}
		m.tempByOptions[bucket] = value
	}
	if cfg.HeadLayers != len(m.layers) || len(cfg.ActCosts)+1 != m.actions {
		return nil, cfg, fmt.Errorf("laya: config/head mismatch")
	}
	return m, cfg, nil
}
