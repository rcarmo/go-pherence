package whisper

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

const maxSpeechConfigJSON = 1 << 20

// speechJSONObject rejects duplicate keys (including nested maps), trailing
// input and excessive nesting before decoding fields. Inputs are caller-owned
// bytes, bounded to 1MiB; no files, network, weights or drivers are opened.
func speechJSONObject(data []byte) (map[string]json.RawMessage, error) {
	if len(data) == 0 || len(data) > maxSpeechConfigJSON {
		return nil, fmt.Errorf("speech config must contain 1..%d bytes", maxSpeechConfigJSON)
	}
	d := json.NewDecoder(bytes.NewReader(data))
	d.UseNumber()
	var walk func(int) error
	walk = func(depth int) error {
		if depth > 32 {
			return fmt.Errorf("speech config nesting exceeds 32")
		}
		token, err := d.Token()
		if err != nil {
			return err
		}
		delimiter, ok := token.(json.Delim)
		if !ok {
			return nil
		}
		switch delimiter {
		case '{':
			seen := map[string]bool{}
			for d.More() {
				key, err := d.Token()
				if err != nil {
					return err
				}
				name, ok := key.(string)
				if !ok || seen[name] {
					return fmt.Errorf("duplicate or invalid JSON key %q", name)
				}
				seen[name] = true
				if err := walk(depth + 1); err != nil {
					return err
				}
			}
			end, err := d.Token()
			if err != nil {
				return err
			}
			if end != json.Delim('}') {
				return fmt.Errorf("invalid JSON object")
			}
		case '[':
			for d.More() {
				if err := walk(depth + 1); err != nil {
					return err
				}
			}
			end, err := d.Token()
			if err != nil {
				return err
			}
			if end != json.Delim(']') {
				return fmt.Errorf("invalid JSON array")
			}
		default:
			return fmt.Errorf("unexpected JSON delimiter")
		}
		return nil
	}
	if err := walk(0); err != nil {
		return nil, fmt.Errorf("invalid speech config: %w", err)
	}
	if _, err := d.Token(); err != io.EOF {
		return nil, fmt.Errorf("trailing speech config input")
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil || fields == nil {
		return nil, fmt.Errorf("speech config must be an object")
	}
	return fields, nil
}

func configField[T any](fields map[string]json.RawMessage, name string) (T, error) {
	var value T
	raw, ok := fields[name]
	if !ok || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return value, fmt.Errorf("missing/null speech config field %s", name)
	}
	// JSON null otherwise silently becomes zero in integer slices/maps (e.g.
	// a suppression token or alignment index). None of these typed fields has
	// nullable members; forced prompt language nulls are handled separately.
	decoder := json.NewDecoder(bytes.NewReader(raw))
	for {
		token, err := decoder.Token()
		if err == io.EOF {
			break
		}
		if err != nil {
			return value, err
		}
		if token == nil {
			return value, fmt.Errorf("null member in speech config field %s", name)
		}
	}
	if err := json.Unmarshal(raw, &value); err != nil {
		return value, fmt.Errorf("invalid speech config field %s: %w", name, err)
	}
	return value, nil
}

func configEqual[T comparable](fields map[string]json.RawMessage, name string, want T, required bool) error {
	if _, ok := fields[name]; !ok && !required {
		return nil
	}
	got, err := configField[T](fields, name)
	if err != nil {
		return err
	}
	if got != want {
		return fmt.Errorf("unsupported speech config %s", name)
	}
	return nil
}

// ParseModelConfigChecked imports HF inference geometry with bounded arithmetic.
// GELU/unscaled embeddings and the standard multilingual token layout are
// required. Unknown fields are rejected. Accepted training, provenance and
// generation-default fields are not executed here;
// generation_config.json is parsed separately and takes precedence. This does
// not certify the checkpoint weights or its model-quality equivalence.
func ParseModelConfigChecked(data []byte) (Config, error) {
	var c Config
	fields, err := speechJSONObject(data)
	if err != nil {
		return c, err
	}
	allowed := strings.Fields("_name_or_path _commit_hash activation_dropout activation_function apply_spec_augment architectures attention_dropout begin_suppress_tokens bos_token_id classifier_proj_size d_model decoder_attention_heads decoder_ffn_dim decoder_layerdrop decoder_layers decoder_start_token_id dropout encoder_attention_heads encoder_ffn_dim encoder_layerdrop encoder_layers eos_token_id forced_decoder_ids init_std is_encoder_decoder max_length max_source_positions max_target_positions model_type num_hidden_layers num_mel_bins pad_token_id scale_embedding suppress_tokens torch_dtype transformers_version use_cache tie_word_embeddings layer_norm_eps is_decoder mask_feature_length mask_feature_min_masks mask_feature_prob mask_time_length mask_time_min_masks mask_time_prob median_filter_width use_weighted_layer_sum vocab_size")
	known := make(map[string]bool, len(allowed))
	for _, name := range allowed {
		known[name] = true
	}
	for name := range fields {
		if !known[name] {
			return Config{}, fmt.Errorf("unsupported model config field %s", name)
		}
	}
	if err := configEqual(fields, "model_type", "whisper", true); err != nil {
		return c, err
	}
	if err := configEqual(fields, "activation_function", "gelu", true); err != nil {
		return c, err
	}
	if err := configEqual(fields, "is_encoder_decoder", true, true); err != nil {
		return c, err
	}
	if err := configEqual(fields, "scale_embedding", false, true); err != nil {
		return c, err
	}
	architectures, err := configField[[]string](fields, "architectures")
	if err != nil {
		return c, err
	}
	if len(architectures) != 1 || architectures[0] != "WhisperForConditionalGeneration" {
		return c, fmt.Errorf("unsupported Whisper architecture")
	}
	for _, item := range []struct {
		name string
		dst  *int
	}{
		{"num_mel_bins", &c.NumMelBins}, {"d_model", &c.EncoderDModel}, {"encoder_layers", &c.EncoderLayers}, {"decoder_layers", &c.DecoderLayers},
		{"encoder_attention_heads", &c.EncoderHeads}, {"decoder_attention_heads", &c.DecoderHeads}, {"encoder_ffn_dim", &c.EncoderFFNDim}, {"decoder_ffn_dim", &c.DecoderFFNDim},
		{"vocab_size", &c.VocabSize}, {"max_target_positions", &c.MaxDecoderLength},
	} {
		value, err := configField[int](fields, item.name)
		if err != nil {
			return Config{}, err
		}
		*item.dst = value
	}
	positions, err := configField[int](fields, "max_source_positions")
	if err != nil {
		return Config{}, err
	}
	if positions < 1 || positions > 1500 {
		return Config{}, fmt.Errorf("invalid source position limit")
	}
	c.MaxLength = 2 * positions
	c.DecoderDModel = c.EncoderDModel
	if c.EncoderDModel < 1 || c.EncoderDModel > 1280 || c.EncoderHeads < 1 || c.EncoderHeads > c.EncoderDModel {
		return Config{}, fmt.Errorf("invalid model/head dimensions")
	}
	c.HeadDim = c.EncoderDModel / c.EncoderHeads
	if err := validatePCMConfig(c); err != nil {
		return Config{}, err
	}
	for name, want := range map[string]int{"bos_token_id": 50257, "eos_token_id": 50257, "pad_token_id": 50257, "decoder_start_token_id": 50258} {
		if err := configEqual(fields, name, want, true); err != nil {
			return Config{}, err
		}
	}
	// These changes affect the inference graph; reject non-default declarations.
	if err := configEqual(fields, "tie_word_embeddings", true, false); err != nil {
		return Config{}, err
	}
	if err := configEqual(fields, "layer_norm_eps", 1e-5, false); err != nil {
		return Config{}, err
	}
	if err := configEqual(fields, "use_cache", true, false); err != nil {
		return Config{}, err
	}
	if err := configEqual(fields, "is_decoder", false, false); err != nil {
		return Config{}, err
	}
	return c, nil
}
