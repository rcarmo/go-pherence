package kv

import (
	"fmt"
	"strings"
)

// ParseCacheTypeBits maps llama.cpp/TurboQuant-style cache type names to the
// bit width used by go-pherence's pure Go KV quantizer. The quantizer remains
// native Go/SIMD; these names are accepted for operational compatibility with
// llama-server presets and runbooks.
func ParseCacheTypeBits(name string) (bits int, enabled bool, err error) {
	switch strings.ToLower(strings.TrimSpace(name)) {
	case "", "f16", "bf16", "f32", "none", "full":
		return 0, false, nil
	case "turbo2", "q2_k", "q2_0":
		return 2, true, nil
	case "turbo3", "q3_k", "q3_0":
		return 3, true, nil
	case "turbo4", "q4_k", "q4_0", "q4_1":
		return 4, true, nil
	case "q5_k", "q5_0", "q5_1":
		return 5, true, nil
	case "q6_k":
		return 6, true, nil
	case "q8_0", "q8_1", "q8":
		return 8, true, nil
	default:
		return 0, false, fmt.Errorf("unsupported KV cache type %q", name)
	}
}

func TurboQuantConfigFromCacheTypes(keyType, valueType string, residualWindow int) (TurboQuantConfig, bool, error) {
	cfg := DefaultTurboQuantConfig()
	if residualWindow >= 0 {
		cfg.ResidualWindow = residualWindow
	}
	kBits, kEnabled, err := ParseCacheTypeBits(keyType)
	if err != nil {
		return cfg, false, err
	}
	vBits, vEnabled, err := ParseCacheTypeBits(valueType)
	if err != nil {
		return cfg, false, err
	}
	if kEnabled {
		cfg.KeyBits = kBits
	}
	if vEnabled {
		cfg.ValueBits = vBits
	}
	return cfg, kEnabled || vEnabled, nil
}
