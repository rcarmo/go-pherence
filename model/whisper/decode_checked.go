package whisper

import (
	"context"
	"errors"
	"fmt"
	"math"
)

var (
	ErrGenerationLimit     = errors.New("Whisper generation reached its token limit without EOT")
	ErrIncompleteTimestamp = errors.New("Whisper text has no closing timestamp")
)

// timestampVocabulary is verified against the caller's tokenizer, not the
// legacy package constants (which match large-v3 but are +1 for older models).
type timestampVocabulary struct {
	sot, eot, language, transcribe, noTimestamps, timestampBegin int
	vocabSize                                                    int
}

func checkedTimestampVocabulary(cfg Config, tokenizer *Tokenizer, language string) (timestampVocabulary, error) {
	var v timestampVocabulary
	if tokenizer == nil || (cfg.VocabSize != 51865 && cfg.VocabSize != 51866) || len(tokenizer.Vocab) != cfg.VocabSize || language == "" {
		return v, fmt.Errorf("checked multilingual decoding requires a complete tokenizer and an explicit language code")
	}
	for id := 0; id < cfg.VocabSize; id++ {
		if _, ok := tokenizer.Vocab[id]; !ok {
			return v, fmt.Errorf("missing tokenizer ID %d", id)
		}
	}
	find := func(text string) (int, error) {
		found := -1
		for id, value := range tokenizer.Vocab {
			if value == text {
				if found >= 0 {
					return 0, fmt.Errorf("duplicate tokenizer control %s", text)
				}
				found = id
			}
		}
		if found < 0 {
			return 0, fmt.Errorf("missing tokenizer control %s", text)
		}
		return found, nil
	}
	for _, field := range []struct {
		name string
		dst  *int
	}{
		{"<|startoftranscript|>", &v.sot}, {"<|endoftext|>", &v.eot},
		{"<|" + language + "|>", &v.language}, {"<|transcribe|>", &v.transcribe},
		{"<|notimestamps|>", &v.noTimestamps}, {"<|0.00|>", &v.timestampBegin},
	} {
		id, err := find(field.name)
		if err != nil {
			return v, err
		}
		*field.dst = id
	}
	// These two multilingual vocabularies share text/EOT/SOT IDs. Reject an
	// incompatible tokenizer instead of letting Tokenizer.Decode skip controls.
	if v.eot != 50257 || v.sot != 50258 || v.language <= v.sot || v.language >= v.transcribe-1 || v.transcribe != cfg.VocabSize-1506 || v.noTimestamps+1 != v.timestampBegin || v.timestampBegin+1501 != cfg.VocabSize {
		return v, fmt.Errorf("incompatible multilingual tokenizer layout")
	}
	for i := 0; i <= 1500; i++ {
		if tokenizer.Vocab[v.timestampBegin+i] != fmt.Sprintf("<|%.2f|>", float64(i)/50) {
			return v, fmt.Errorf("invalid timestamp token at index %d", i)
		}
	}
	v.vocabSize = cfg.VocabSize
	return v, nil
}

// checkedTimestampMask follows Transformers 4.57.1's Whisper timestamp pairing,
// monotonicity, initial bound and aggregate timestamp probability rules. Values
// are raw logits: logsumexp(timestamp logits)>max(text logits) is equivalent to
// comparing log probabilities because their normalisation constant cancels.
func checkedTimestampMask(logits []float32, generated []int, v timestampVocabulary, maxInitial int) {
	mask := func(start, end int) {
		for i := start; i < end; i++ {
			logits[i] = float32(math.Inf(-1))
		}
	}
	mask(v.sot, v.timestampBegin) // language/task/control tokens are prompt-only
	if len(generated) == 0 {
		mask(0, v.timestampBegin)
		mask(v.timestampBegin+maxInitial+1, len(logits))
	} else {
		lastTimestamp := generated[len(generated)-1] >= v.timestampBegin
		penultimateTimestamp := len(generated) < 2 || generated[len(generated)-2] >= v.timestampBegin
		if lastTimestamp {
			if penultimateTimestamp {
				mask(v.timestampBegin, len(logits))
			} else {
				mask(0, v.eot)
			}
		}
		for i := len(generated) - 1; i >= 0; i-- {
			if generated[i] >= v.timestampBegin {
				minimum := generated[i] + 1
				if lastTimestamp && !penultimateTimestamp {
					minimum--
				}
				mask(v.timestampBegin, minimum)
				break
			}
		}
	}
	tsMax, textMax := math.Inf(-1), math.Inf(-1)
	for i, score := range logits {
		if i >= v.timestampBegin {
			tsMax = math.Max(tsMax, float64(score))
		} else {
			textMax = math.Max(textMax, float64(score))
		}
	}
	if !math.IsInf(tsMax, -1) {
		sum := 0.0
		for _, score := range logits[v.timestampBegin:] {
			sum += math.Exp(float64(score) - tsMax)
		}
		if tsMax+math.Log(sum) > textMax {
			mask(0, v.timestampBegin)
		}
	}
}

// decodeCheckedTimestamps is separated from tensor execution for deterministic
// token/logit control-flow tests. Such tests do not qualify real-model output.
func decodeCheckedTimestamps(ctx context.Context, cfg Config, tokenizer *Tokenizer, v timestampVocabulary, opts PCMTranscribeOptions, suppress, beginSuppress []int, forward func(int) ([]float32, error)) ([]Segment, error) {
	for _, ids := range [][]int{suppress, beginSuppress} {
		for _, id := range ids {
			if id < 0 || id >= v.vocabSize {
				return nil, fmt.Errorf("invalid suppression token %d", id)
			}
		}
	}
	maxTokens := opts.MaxNewTokens
	if maxTokens == 0 {
		maxTokens = cfg.MaxDecoderLength - 3
	}
	if maxTokens < 1 || maxTokens > cfg.MaxDecoderLength-3 || opts.MaxInitialTimestampIndex < 0 || opts.MaxInitialTimestampIndex > 1500 {
		return nil, fmt.Errorf("invalid checked generation bounds")
	}
	step := func(token int) ([]float32, error) {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		logits, err := forward(token)
		if err != nil {
			return nil, err
		}
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if len(logits) != v.vocabSize {
			return nil, fmt.Errorf("invalid decoder logit shape")
		}
		for _, score := range logits {
			if math.IsNaN(float64(score)) || math.IsInf(float64(score), 1) {
				return nil, fmt.Errorf("non-finite decoder logits")
			}
		}
		return logits, nil
	}
	var logits []float32
	var err error
	for _, token := range []int{v.sot, v.language, v.transcribe} {
		logits, err = step(token)
		if err != nil {
			return nil, err
		}
	}
	generated := make([]int, 0, maxTokens)
	var segments []Segment
	var text []int
	start := -1
	for i := 0; i < maxTokens; i++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		for _, id := range suppress {
			logits[id] = float32(math.Inf(-1))
		}
		if i == 0 {
			for _, id := range beginSuppress {
				logits[id] = float32(math.Inf(-1))
			}
		}
		checkedTimestampMask(logits, generated, v, opts.MaxInitialTimestampIndex)
		next := argmax(logits)
		if math.IsInf(float64(logits[next]), -1) {
			return nil, fmt.Errorf("all decoder tokens are suppressed")
		}
		if next == v.eot {
			if len(text) != 0 {
				return nil, ErrIncompleteTimestamp
			}
			return segments, nil
		}
		if next >= v.timestampBegin {
			end := next - v.timestampBegin
			if len(text) != 0 {
				if start < 0 || end <= start {
					return nil, fmt.Errorf("invalid segment timestamp order")
				}
				segments = append(segments, Segment{Start: float64(start) / 50, End: float64(end) / 50, Text: tokenizer.Decode(text), Tokens: text})
				text = nil
			}
			start = end
		} else {
			if start < 0 || next >= v.eot {
				return nil, fmt.Errorf("unexpected token outside timestamp segment")
			}
			text = append(text, next)
		}
		generated = append(generated, next)
		// Do not consume a position after the final allowed token. Prompt tokens
		// count towards MaxDecoderLength, unlike the legacy loop's bound.
		if i+1 < maxTokens {
			logits, err = step(next)
			if err != nil {
				return nil, err
			}
		}
	}
	return nil, ErrGenerationLimit
}
