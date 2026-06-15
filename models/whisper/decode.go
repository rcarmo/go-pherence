package whisper

import "strings"

// Special token IDs for Whisper
const (
	TokenSOT            = 50258 // <|startoftranscript|>
	TokenEOT            = 50257 // <|endoftext|>
	TokenTranslate      = 50359 // <|translate|>
	TokenTranscribe     = 50360 // <|transcribe|>
	TokenNoTimestamps   = 50364 // <|notimestamps|>
	TokenEnglish        = 50259 // <|en|>
	TokenTimestampBegin = 50365 // <|0.00|>
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

// GreedyDecode performs greedy autoregressive no-timestamps transcription in English.
func GreedyDecode(dec *Decoder, state *DecoderState, cfg Config) []int {
	return GreedyDecodePrompt(dec, state, cfg, TokenEnglish, TokenTranscribe)
}

// GreedyDecodePrompt performs greedy autoregressive no-timestamps decoding with
// an explicit Whisper language token and task token (transcribe or translate).
func GreedyDecodePrompt(dec *Decoder, state *DecoderState, cfg Config, languageToken, taskToken int) []int {
	maxTokens := cfg.MaxDecoderLength
	if languageToken == 0 {
		languageToken = TokenEnglish
	}
	if taskToken == 0 {
		taskToken = TokenTranscribe
	}

	// Start with SOT + language + task + notimestamps.
	prompt := []int{TokenSOT, languageToken, taskToken, TokenNoTimestamps}

	// Feed prompt tokens once. The logits returned for the final prompt token are
	// the distribution for the first generated token; do not feed the final
	// prompt token a second time, or absolute positions/KV cache drift from the
	// intended Whisper prompt.
	var logits []float32
	for _, tok := range prompt {
		logits = dec.ForwardToken(tok, state)
	}

	// Generate tokens greedily
	tokens := make([]int, 0, maxTokens)
	for i := 0; i < maxTokens; i++ {
		suppressNonTextSpecials(logits)
		suppressTokenIDs(logits, dec.SuppressTokens)
		if i == 0 {
			suppressTokenIDs(logits, dec.BeginSuppressTokens)
		}
		suppressRecentRepeats(logits, tokens, 6)
		if i == 0 && TokenEOT < len(logits) {
			// Avoid an immediate empty transcript on short clips where the blank/EOT
			// prior can dominate the first greedy step.
			logits[TokenEOT] = -1e30
		}
		nextTok := argmax(logits)

		if nextTok == TokenEOT || wouldRepeatRun(tokens, nextTok, 6) || repeatedNGram(tokens, nextTok, 3) {
			break
		}
		tokens = append(tokens, nextTok)
		logits = dec.ForwardToken(nextTok, state)
	}

	return tokens
}

// GreedyDecodeWithTimestamps performs greedy decoding and returns timed segments.
func GreedyDecodeWithTimestamps(dec *Decoder, state *DecoderState, cfg Config) []Segment {
	maxTokens := cfg.MaxDecoderLength

	// Start with SOT + language + task (allow timestamps)
	prompt := []int{TokenSOT, TokenEnglish, TokenTranscribe}

	var segments []Segment
	var currentTokens []int
	var startTime float64

	var logits []float32
	for _, tok := range prompt {
		logits = dec.ForwardToken(tok, state)
	}
	flushCurrent := func(end float64) {
		if len(currentTokens) == 0 {
			return
		}
		segments = append(segments, Segment{
			Start:  startTime,
			End:    end,
			Text:   TokensToText(currentTokens),
			Tokens: currentTokens,
		})
		currentTokens = nil
	}
	for i := 0; i < maxTokens; i++ {
		nextTok := argmax(logits)

		if nextTok == TokenEOT {
			flushCurrent(startTime + 30.0)
			break
		}

		if IsTimestamp(nextTok) {
			t := TimestampToSeconds(nextTok)
			flushCurrent(t)
			startTime = t
		} else {
			currentTokens = append(currentTokens, nextTok)
		}

		logits = dec.ForwardToken(nextTok, state)
	}

	flushCurrent(startTime + 30.0)
	return segments
}

func suppressTokenIDs(logits []float32, ids []int) {
	for _, tok := range ids {
		if tok >= 0 && tok < len(logits) {
			logits[tok] = -1e30
		}
	}
}

func suppressRecentRepeats(logits []float32, tokens []int, window int) {
	start := len(tokens) - window
	if start < 0 {
		start = 0
	}
	for _, tok := range tokens[start:] {
		if tok >= 0 && tok < len(logits) {
			logits[tok] -= 8.0
		}
	}
}

func wouldRepeatRun(tokens []int, nextTok, maxRun int) bool {
	if maxRun <= 1 || len(tokens) < maxRun-1 {
		return false
	}
	for i := len(tokens) - (maxRun - 1); i < len(tokens); i++ {
		if tokens[i] != nextTok {
			return false
		}
	}
	return true
}

func repeatedNGram(tokens []int, nextTok, n int) bool {
	if n <= 1 || len(tokens) < 2*n-1 {
		return false
	}
	candidateStart := len(tokens) - (n - 1)
	for start := 0; start+n <= candidateStart; start++ {
		match := true
		for i := 0; i < n-1; i++ {
			if tokens[start+i] != tokens[candidateStart+i] {
				match = false
				break
			}
		}
		if match && tokens[start+n-1] == nextTok {
			return true
		}
	}
	return false
}

func suppressNonTextSpecials(logits []float32) {
	// Whisper suppresses prompt/control/language/timestamp tokens during
	// no-timestamps transcription. Without this, greedy decoding can loop on
	// <|translate|>, <|notimestamps|>, or language tags.
	for tok := TokenSOT; tok < len(logits); tok++ {
		logits[tok] = -1e30
	}
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

// DefaultTokenizer is the global tokenizer instance (set by LoadTokenizerGlobal).
var DefaultTokenizer *Tokenizer

// LoadTokenizerGlobal loads a tokenizer and sets it as the default for TokensToText.
func LoadTokenizerGlobal(path string) error {
	t, err := LoadTokenizer(path)
	if err != nil {
		return err
	}
	DefaultTokenizer = t
	return nil
}

// TokensToText converts token IDs to text using the loaded tokenizer.
func TokensToText(tokens []int) string {
	if DefaultTokenizer != nil {
		return DefaultTokenizer.Decode(tokens)
	}
	// Fallback: placeholder
	var parts []string
	for _, tok := range tokens {
		if tok >= TokenSOT || tok < 0 {
			continue
		}
		parts = append(parts, "[tok]")
	}
	return strings.Join(parts, "")
}
