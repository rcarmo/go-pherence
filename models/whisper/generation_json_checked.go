package whisper

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

// CheckedGenerationConfig is immutable after construction. Its private fields
// prevent bypassing parser validation; it can be shared by checked PCM calls.
// It describes explicit-language greedy transcription with timestamps, not the
// full Transformers generation feature set. No model/decoder globals are changed.
type CheckedGenerationConfig struct {
	cfg                     Config
	vocabulary              timestampVocabulary
	languages               map[string]int
	suppress, beginSuppress []int
	alignmentHeads          [][2]int
	maxLength, maxInitial   int
}

// ParseGenerationConfigChecked validates HF generation metadata against the
// model config and full tokenizer. Explicit per-call language + transcribe +
// timestamps override forced_decoder_ids and return_timestamps. Those source
// defaults are validated but never appended to the prompt. Unknown generation
// controls and non-default sampling/beam/repetition modes are rejected. Alignment
// heads are validated and retained for the separate checked word-timing path.
func ParseGenerationConfigChecked(data []byte, cfg Config, tokenizer *Tokenizer) (*CheckedGenerationConfig, error) {
	if err := validatePCMConfig(cfg); err != nil {
		return nil, err
	}
	fields, err := speechJSONObject(data)
	if err != nil {
		return nil, err
	}
	allowed := strings.Fields("_from_model_config transformers_version alignment_heads begin_suppress_tokens bos_token_id decoder_start_token_id eos_token_id forced_decoder_ids is_multilingual lang_to_id max_initial_timestamp_index max_length no_timestamps_token_id pad_token_id prev_sot_token_id return_timestamps suppress_tokens task_to_id use_scan do_sample num_beams num_beam_groups num_return_sequences temperature top_k top_p typical_p repetition_penalty encoder_repetition_penalty no_repeat_ngram_size encoder_no_repeat_ngram_size length_penalty min_length min_new_tokens early_stopping use_cache renormalize_logits remove_invalid_values output_attentions output_hidden_states output_scores return_dict_in_generate language task")
	known := make(map[string]bool, len(allowed))
	for _, key := range allowed {
		known[key] = true
	}
	for key := range fields {
		if !known[key] {
			return nil, fmt.Errorf("unsupported generation field %s", key)
		}
	}
	v, err := checkedTimestampVocabulary(cfg, tokenizer, "en")
	if err != nil {
		return nil, err
	}
	for name, want := range map[string]int{"bos_token_id": v.eot, "eos_token_id": v.eot, "pad_token_id": v.eot, "decoder_start_token_id": v.sot, "no_timestamps_token_id": v.noTimestamps} {
		if err := configEqual(fields, name, want, true); err != nil {
			return nil, err
		}
	}
	if err := configEqual(fields, "is_multilingual", true, true); err != nil {
		return nil, err
	}
	if err := configEqual(fields, "prev_sot_token_id", v.noTimestamps-2, false); err != nil {
		return nil, err
	}
	for name, want := range map[string]bool{"do_sample": false, "early_stopping": false, "use_cache": true, "renormalize_logits": false, "remove_invalid_values": false, "output_attentions": false, "output_hidden_states": false, "output_scores": false, "return_dict_in_generate": false} {
		if err := configEqual(fields, name, want, false); err != nil {
			return nil, err
		}
	}
	for name, want := range map[string]int{"num_beams": 1, "num_beam_groups": 1, "num_return_sequences": 1, "top_k": 50, "no_repeat_ngram_size": 0, "encoder_no_repeat_ngram_size": 0, "min_length": 0, "min_new_tokens": 0} {
		if err := configEqual(fields, name, want, false); err != nil {
			return nil, err
		}
	}
	for name, want := range map[string]float64{"temperature": 1, "top_p": 1, "typical_p": 1, "repetition_penalty": 1, "encoder_repetition_penalty": 1, "length_penalty": 1} {
		if err := configEqual(fields, name, want, false); err != nil {
			return nil, err
		}
	}
	// Temperature 1 is the HF default without sampling; actual selection is argmax.
	for _, key := range []string{"_from_model_config", "return_timestamps"} {
		if _, ok := fields[key]; ok {
			if _, err := configField[bool](fields, key); err != nil {
				return nil, err
			}
		}
	}
	if _, ok := fields["transformers_version"]; ok {
		if _, err := configField[string](fields, "transformers_version"); err != nil {
			return nil, err
		}
	}
	if err := configEqual(fields, "use_scan", false, false); err != nil {
		return nil, err
	}
	maxLength, err := configField[int](fields, "max_length")
	if err != nil {
		return nil, err
	}
	if maxLength < 4 || maxLength > cfg.MaxDecoderLength {
		return nil, fmt.Errorf("invalid generation position limit")
	}
	maxInitial, err := configField[int](fields, "max_initial_timestamp_index")
	if err != nil {
		return nil, err
	}
	if maxInitial < 0 || maxInitial > 1500 {
		return nil, fmt.Errorf("invalid initial timestamp bound")
	}
	tasks, err := configField[map[string]int](fields, "task_to_id")
	if err != nil {
		return nil, err
	}
	if len(tasks) != 2 || tasks["transcribe"] != v.transcribe || tasks["translate"] != v.transcribe-1 {
		return nil, fmt.Errorf("invalid generation task tokens")
	}
	langs, err := configField[map[string]int](fields, "lang_to_id")
	if err != nil {
		return nil, err
	}
	if len(langs) != v.transcribe-v.sot-2 {
		return nil, fmt.Errorf("incomplete language token map")
	}
	languages := make(map[string]int, len(langs))
	seen := make(map[int]bool, len(langs))
	for name, id := range langs {
		if !strings.HasPrefix(name, "<|") || !strings.HasSuffix(name, "|>") || len(name) <= 4 || id <= v.sot || id >= v.transcribe-1 || tokenizer.Vocab[id] != name || seen[id] {
			return nil, fmt.Errorf("invalid language map entry")
		}
		seen[id] = true
		languages[name[2:len(name)-2]] = id
	}
	if _, ok := fields["language"]; ok {
		language, err := configField[string](fields, "language")
		if err != nil {
			return nil, err
		}
		if _, ok := languages[language]; !ok {
			return nil, fmt.Errorf("invalid source language default")
		}
	}
	if _, ok := fields["task"]; ok {
		task, err := configField[string](fields, "task")
		if err != nil {
			return nil, err
		}
		if task != "transcribe" && task != "translate" {
			return nil, fmt.Errorf("invalid source task default")
		}
	}
	if raw, ok := fields["forced_decoder_ids"]; ok {
		var pairs [][]json.RawMessage
		if err := json.Unmarshal(raw, &pairs); err != nil || len(pairs) > 3 {
			return nil, fmt.Errorf("invalid forced prompt defaults")
		}
		seenPosition := map[int]bool{}
		for _, pair := range pairs {
			if len(pair) != 2 {
				return nil, fmt.Errorf("invalid forced prompt pair")
			}
			var pos int
			if err := json.Unmarshal(pair[0], &pos); err != nil || pos < 1 || pos > 3 || seenPosition[pos] {
				return nil, fmt.Errorf("invalid forced prompt position")
			}
			seenPosition[pos] = true
			var id *int
			if err := json.Unmarshal(pair[1], &id); err != nil {
				return nil, err
			}
			if id == nil {
				if pos != 1 {
					return nil, fmt.Errorf("null forced token outside language position")
				}
				continue
			}
			if (pos == 1 && !seen[*id]) || (pos == 2 && *id != v.transcribe && *id != v.transcribe-1) || (pos == 3 && *id != v.noTimestamps) {
				return nil, fmt.Errorf("incompatible forced prompt default")
			}
		}
	}
	var alignmentHeads [][2]int
	if _, ok := fields["alignment_heads"]; ok {
		heads, err := configField[[][]int](fields, "alignment_heads")
		if err != nil {
			return nil, err
		}
		if len(heads) > cfg.DecoderLayers*cfg.DecoderHeads {
			return nil, fmt.Errorf("too many alignment heads")
		}
		seenHeads := make(map[[2]int]bool, len(heads))
		for _, head := range heads {
			if len(head) != 2 || head[0] < 0 || head[0] >= cfg.DecoderLayers || head[1] < 0 || head[1] >= cfg.DecoderHeads {
				return nil, fmt.Errorf("invalid alignment head")
			}
			pair := [2]int{head[0], head[1]}
			if seenHeads[pair] {
				return nil, fmt.Errorf("duplicate alignment head")
			}
			seenHeads[pair] = true
			alignmentHeads = append(alignmentHeads, pair)
		}
	}
	suppress, err := configField[[]int](fields, "suppress_tokens")
	if err != nil {
		return nil, err
	}
	begin, err := configField[[]int](fields, "begin_suppress_tokens")
	if err != nil {
		return nil, err
	}
	if err := validateSuppressionLists(v, suppress, begin); err != nil {
		return nil, err
	}
	for _, id := range suppress {
		if id == v.eot || id >= v.timestampBegin {
			return nil, fmt.Errorf("suppression prevents checked timestamp decoding")
		}
	}
	for _, id := range begin {
		if id >= v.timestampBegin {
			return nil, fmt.Errorf("initial timestamp suppression unsupported")
		}
	}
	return &CheckedGenerationConfig{cfg: cfg, vocabulary: v, languages: languages, suppress: suppress, beginSuppress: begin, alignmentHeads: alignmentHeads, maxLength: maxLength, maxInitial: maxInitial}, nil
}

// SupportsWordAlignment reports whether this checked immutable policy carries
// at least one validated alignment head for the same model/tokenizer contract.
func (g *CheckedGenerationConfig) SupportsWordAlignment() bool {
	return g != nil && len(g.alignmentHeads) > 0
}

func validateSuppressionLists(v timestampVocabulary, lists ...[]int) error {
	for _, ids := range lists {
		if len(ids) > v.vocabSize {
			return fmt.Errorf("oversized suppression list")
		}
		for _, id := range ids {
			if id < 0 || id >= v.vocabSize {
				return fmt.Errorf("invalid suppression token %d", id)
			}
		}
	}
	return nil
}

// resolvePCMGeneration snapshots per-call policy. A supplied config owns the
// initial timestamp bound; MaxNewTokens may only shorten its position budget.
func resolvePCMGeneration(cfg Config, v timestampVocabulary, opts PCMTranscribeOptions, suppress, begin []int) (PCMTranscribeOptions, []int, []int, error) {
	if opts.Generation != nil {
		g := opts.Generation
		if g.cfg != cfg || g.maxLength < 4 || g.maxLength > cfg.MaxDecoderLength || g.vocabulary.sot != v.sot || g.vocabulary.eot != v.eot || g.vocabulary.transcribe != v.transcribe || g.vocabulary.timestampBegin != v.timestampBegin || g.languages[opts.Language] != v.language {
			return opts, nil, nil, fmt.Errorf("generation config/model/tokenizer mismatch")
		}
		if opts.MaxInitialTimestampIndex != 0 {
			return opts, nil, nil, fmt.Errorf("generation config owns initial timestamp bound")
		}
		if opts.MaxNewTokens < 0 || opts.MaxNewTokens > g.maxLength-3 {
			return opts, nil, nil, fmt.Errorf("generation token request exceeds config limit")
		}
		if opts.MaxNewTokens == 0 {
			opts.MaxNewTokens = g.maxLength - 3
		}
		opts.MaxInitialTimestampIndex = g.maxInitial
		suppress, begin = g.suppress, g.beginSuppress
	}
	if err := validateSuppressionLists(v, suppress, begin); err != nil {
		return opts, nil, nil, err
	}
	return opts, suppress, begin, nil
}

// LoadConfiguredModelSourceChecked validates both metadata documents and the
// tokenizer BEFORE reading tensor payloads. It returns an owned model and an
// immutable generation policy. Pass that policy as PCMTranscribeOptions.Generation.
// The source remains caller-owned. This performs no inference or GPU allocation.
func LoadConfiguredModelSourceChecked(ctx context.Context, source CheckedTensorSource, modelJSON, generationJSON []byte, tokenizer *Tokenizer) (*Whisper, *CheckedGenerationConfig, error) {
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	cfg, err := ParseModelConfigChecked(modelJSON)
	if err != nil {
		return nil, nil, err
	}
	generation, err := ParseGenerationConfigChecked(generationJSON, cfg, tokenizer)
	if err != nil {
		return nil, nil, err
	}
	w, err := LoadModelSourceChecked(ctx, source, cfg)
	if err != nil {
		return nil, nil, err
	}
	return w, generation, nil
}
