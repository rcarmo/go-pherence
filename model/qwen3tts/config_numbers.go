package qwen3tts

import (
	"encoding/json"
	"fmt"
	"strconv"
)

// The planner materialises one index per group (published checkpoints use 16).
// This generous metadata bound is not a whole-model memory budget.
const maxCodeGroups = 1024

// Missing known fields retain published defaults; present malformed fields must
// not silently become defaults. Unknown upstream metadata is deliberately kept
// compatible. UseNumber preserves integers beyond float64's exact range.
func validateNumericMetadata(root map[string]any) error {
	object := func(parent map[string]any, key string) (map[string]any, error) {
		v, exists := parent[key]
		if !exists {
			return nil, nil
		}
		m, ok := v.(map[string]any)
		if !ok || m == nil {
			return nil, fmt.Errorf("Qwen3-TTS %s must be an object", key)
		}
		return m, nil
	}
	talker, err := object(root, "talker_config")
	if err != nil {
		return err
	}
	cp, err := object(talker, "code_predictor_config")
	if err != nil {
		return err
	}
	speaker, err := object(root, "speaker_encoder_config")
	if err != nil {
		return err
	}
	rope, err := object(talker, "rope_scaling")
	if err != nil {
		return err
	}
	integer := func(v any) bool {
		n, ok := v.(json.Number)
		if !ok {
			return false
		}
		_, err := strconv.ParseInt(string(n), 10, strconv.IntSize)
		return err == nil
	}
	for _, scope := range []struct {
		m    map[string]any
		keys []string
	}{
		{talker, []string{"hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size", "text_vocab_size", "text_hidden_size", "max_position_embeddings"}},
		{cp, []string{"hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size", "num_code_groups"}},
		{speaker, []string{"enc_dim", "sample_rate"}},
	} {
		for _, key := range scope.keys {
			if v, exists := scope.m[key]; exists && !integer(v) {
				return fmt.Errorf("Qwen3-TTS %s must be an in-range integer", key)
			}
		}
	}
	for _, m := range []map[string]any{talker, cp} {
		for _, key := range []string{"rms_norm_eps", "rope_theta"} {
			if v, exists := m[key]; exists {
				n, ok := v.(json.Number)
				if !ok {
					return fmt.Errorf("Qwen3-TTS %s must be numeric", key)
				}
				f, err := n.Float64()
				if err != nil || !finitePositive(f) {
					return fmt.Errorf("Qwen3-TTS %s must be finite and positive", key)
				}
			}
		}
	}
	if v, exists := rope["mrope_section"]; exists {
		a, ok := v.([]any)
		if !ok || len(a) != 3 {
			return fmt.Errorf("Qwen3-TTS mrope_section must contain three integers")
		}
		for _, v := range a {
			if !integer(v) {
				return fmt.Errorf("Qwen3-TTS mrope_section must contain in-range integers")
			}
		}
	}
	return nil
}
