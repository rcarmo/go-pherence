package model

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
)

// REAPConfig describes static expert pruning for MoE models. It is intentionally
// runtime-local and pure Go: it constrains router/top-k selection without relying
// on llama.cpp or external pruning code.
type REAPConfig struct {
	Enabled            bool                 `json:"enabled"`
	PruneRatio         float64              `json:"prune_ratio,omitempty"`
	DefaultActive      []int                `json:"active_experts,omitempty"`
	LayerActive        map[string][]int     `json:"layers,omitempty"`
	LayerActiveNumeric map[int]map[int]bool `json:"-"`
	DefaultMask        map[int]bool         `json:"-"`
}

func InferREAPConfigFromName(name string) *REAPConfig {
	re := regexp.MustCompile(`(?i)reap[-_ ]?(\d{1,2})(?:\D|$)`)
	m := re.FindStringSubmatch(name)
	if len(m) < 2 {
		return nil
	}
	n, err := strconv.Atoi(m[1])
	if err != nil || n <= 0 || n >= 100 {
		return nil
	}
	cfg := &REAPConfig{Enabled: true, PruneRatio: float64(n) / 100.0}
	if err := cfg.normalize(); err != nil {
		return nil
	}
	return cfg
}

func LoadREAPConfig(dir string) (*REAPConfig, error) {
	for _, name := range []string{"reap_config.json", "reap.json"} {
		path := filepath.Join(dir, name)
		data, err := os.ReadFile(path)
		if err != nil {
			if os.IsNotExist(err) {
				continue
			}
			return nil, err
		}
		var cfg REAPConfig
		if err := json.Unmarshal(data, &cfg); err != nil {
			return nil, fmt.Errorf("parse %s: %w", path, err)
		}
		if err := cfg.normalize(); err != nil {
			return nil, fmt.Errorf("parse %s: %w", path, err)
		}
		return &cfg, nil
	}
	return InferREAPConfigFromName(filepath.Base(dir)), nil
}

func (r *REAPConfig) normalize() error {
	if r == nil {
		return nil
	}
	r.Enabled = true
	if r.PruneRatio < 0 || r.PruneRatio >= 1 || math.IsNaN(r.PruneRatio) {
		return fmt.Errorf("invalid prune_ratio=%g", r.PruneRatio)
	}
	r.DefaultMask = make(map[int]bool, len(r.DefaultActive))
	for _, id := range r.DefaultActive {
		if id < 0 {
			return fmt.Errorf("negative active expert %d", id)
		}
		r.DefaultMask[id] = true
	}
	r.LayerActiveNumeric = make(map[int]map[int]bool, len(r.LayerActive))
	for key, ids := range r.LayerActive {
		var layer int
		if _, err := fmt.Sscanf(key, "%d", &layer); err != nil || layer < 0 {
			return fmt.Errorf("invalid layer key %q", key)
		}
		mask := make(map[int]bool, len(ids))
		for _, id := range ids {
			if id < 0 {
				return fmt.Errorf("negative active expert %d in layer %d", id, layer)
			}
			mask[id] = true
		}
		r.LayerActiveNumeric[layer] = mask
	}
	return nil
}

func (r *REAPConfig) Allows(layer, expert int) bool {
	if r == nil || !r.Enabled {
		return true
	}
	if mask, ok := r.LayerActiveNumeric[layer]; ok && len(mask) > 0 {
		return mask[expert]
	}
	if len(r.DefaultMask) > 0 {
		return r.DefaultMask[expert]
	}
	return true
}
