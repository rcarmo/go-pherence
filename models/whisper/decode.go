package whisper

import "strings"

// Special token IDs for Whisper
const (
	TokenSOT            = 50258 // <|startoftranscript|>
	TokenEOT            = 50257 // <|endoftext|>
	TokenTranscribe     = 50359 // <|transcribe|>
	TokenTranslate      = 50358 // <|translate|>
	TokenNoTimestamps   = 50363 // <|notimestamps|>
	TokenEnglish        = 50259 // <|en|>
	TokenTimestampBegin = 50364 // <|0.00|>
)

// IsTimestamp returns true if the token is a timestamp token.
func IsTimestamp(tok int) bool {
	return tok >= TokenTimestampBegin
}

// TimestampToSeconds converts a timestamp token to seconds.
// Each timestamp token represents 0.02s increments.
func TimestampToSeconds(tok int) float64 {
	if tok < TokenTimestampBegin {
		return 0
	}
	return float64(tok-TokenTimestampBegin) * 0.02
}

// Segment represents a transcribed audio segment with timing.
type Segment struct {
	Start  float64 // seconds
	End    float64 // seconds
	Text   string
	Tokens []int
}

// GreedyDecode performs greedy autoregressive decoding.
func GreedyDecode(dec *Decoder, state *DecoderState, cfg Config) []int {
	maxTokens := cfg.MaxDecoderLength

	// Start with SOT + language + task + notimestamps
	prompt := []int{TokenSOT, TokenEnglish, TokenTranscribe, TokenNoTimestamps}

	// Feed prompt tokens
	for _, tok := range prompt {
		dec.ForwardToken(tok, state)
	}

	// Generate tokens greedily
	tokens := make([]int, 0, maxTokens)
	for i := 0; i < maxTokens; i++ {
		var prevTok int
		if len(tokens) == 0 {
			prevTok = prompt[len(prompt)-1]
		} else {
			prevTok = tokens[len(tokens)-1]
		}

		logits := dec.ForwardToken(prevTok, state)
		nextTok := argmax(logits)

		if nextTok == TokenEOT {
			break
		}
		tokens = append(tokens, nextTok)
	}

	return tokens
}

// GreedyDecodeWithTimestamps performs greedy decoding and returns timed segments.
func GreedyDecodeWithTimestamps(dec *Decoder, state *DecoderState, cfg Config) []Segment {
	maxTokens := cfg.MaxDecoderLength

	// Start with SOT + language + task (allow timestamps)
	prompt := []int{TokenSOT, TokenEnglish, TokenTranscribe}
	for _, tok := range prompt {
		dec.ForwardToken(tok, state)
	}

	var segments []Segment
	var currentTokens []int
	var startTime float64

	prevTok := prompt[len(prompt)-1]
	for i := 0; i < maxTokens; i++ {
		logits := dec.ForwardToken(prevTok, state)
		nextTok := argmax(logits)

		if nextTok == TokenEOT {
			// Flush remaining segment
			if len(currentTokens) > 0 {
				segments = append(segments, Segment{
					Start:  startTime,
					End:    startTime + 30.0, // end of chunk
					Text:   TokensToText(currentTokens),
					Tokens: currentTokens,
				})
			}
			break
		}

		if IsTimestamp(nextTok) {
			t := TimestampToSeconds(nextTok)
			if len(currentTokens) > 0 {
				// End of a segment
				segments = append(segments, Segment{
					Start:  startTime,
					End:    t,
					Text:   TokensToText(currentTokens),
					Tokens: currentTokens,
				})
				currentTokens = nil
			}
			startTime = t
		} else {
			currentTokens = append(currentTokens, nextTok)
		}

		prevTok = nextTok
	}

	return segments
}

// argmax returns the index of the maximum value.
func argmax(x []float32) int {
	if len(x) == 0 {
		return 0
	}
	maxIdx := 0
	maxVal := x[0]
	for i, v := range x[1:] {
		if v > maxVal {
			maxVal = v
			maxIdx = i + 1
		}
	}
	return maxIdx
}

// TokensToText is a placeholder that converts token IDs to text.
// A proper implementation would use the Whisper tokenizer (GPT-2 BPE).
func TokensToText(tokens []int) string {
	// Filter out special tokens and timestamps
	var parts []string
	for _, tok := range tokens {
		if tok >= TokenSOT || tok < 0 {
			continue // Skip special tokens
		}
		// Placeholder: represent as token ID
		parts = append(parts, tokenToString(tok))
	}
	return strings.Join(parts, "")
}

// tokenToString is a placeholder — real implementation needs the Whisper BPE vocab.
func tokenToString(tok int) string {
	// This would look up the token in the vocabulary
	// For now just return a placeholder
	return "[tok]"
}
