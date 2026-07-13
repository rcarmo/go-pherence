package mosstranscribe

import (
	"fmt"
	"path/filepath"
	"strings"

	basetokenizer "github.com/rcarmo/go-pherence/loader/tokenizer"
)

const AudioPadToken = "<|audio_pad|>"

// Processor owns the checkpoint tokenizer contract needed for prompt expansion
// and generated transcript decoding.
type Processor struct {
	Tokenizer    *basetokenizer.Tokenizer
	AudioTokenID int
	DigitTokenID [10]int
}

func LoadProcessor(modelDir string, expectedAudioTokenID int) (*Processor, error) {
	tok, err := basetokenizer.Load(filepath.Join(modelDir, "tokenizer.json"))
	if err != nil {
		return nil, fmt.Errorf("MOSS tokenizer: %w", err)
	}
	return NewProcessor(tok, expectedAudioTokenID)
}

func NewProcessor(tok *basetokenizer.Tokenizer, expectedAudioTokenID int) (*Processor, error) {
	if tok == nil {
		return nil, fmt.Errorf("MOSS tokenizer: nil tokenizer")
	}
	audioTokenID, ok := tok.Vocab[AudioPadToken]
	if !ok || audioTokenID != expectedAudioTokenID {
		return nil, fmt.Errorf("MOSS tokenizer: %s id=%d present=%v, want %d", AudioPadToken, audioTokenID, ok, expectedAudioTokenID)
	}
	processor := &Processor{Tokenizer: tok, AudioTokenID: audioTokenID}
	for digit := 0; digit < 10; digit++ {
		ids := tok.Encode(string(rune('0' + digit)))
		if len(ids) != 1 {
			return nil, fmt.Errorf("MOSS tokenizer: digit %d encodes as %v, want one token", digit, ids)
		}
		processor.DigitTokenID[digit] = ids[0]
	}
	return processor, nil
}

// EncodePrompt replaces exactly one audio placeholder with audio feature IDs
// and time markers, matching MossTranscribeDiarizeProcessor._expand_audio_token.
func (p *Processor) EncodePrompt(prompt string, audioTokens, maxLength int) ([]int, error) {
	if p == nil || p.Tokenizer == nil {
		return nil, fmt.Errorf("MOSS tokenizer: nil processor")
	}
	if strings.Count(prompt, AudioPadToken) != 1 {
		return nil, fmt.Errorf("MOSS tokenizer: expected exactly one %s placeholder", AudioPadToken)
	}
	before, after, _ := strings.Cut(prompt, AudioPadToken)
	span := AudioSpanIDs(audioTokens, p.AudioTokenID, p.DigitTokenID, TimeMarkerEverySecond)
	ids := make([]int, 0, len(span)+32)
	ids = append(ids, p.Tokenizer.Encode(before)...)
	ids = append(ids, span...)
	ids = append(ids, p.Tokenizer.Encode(after)...)
	if maxLength <= 0 || len(ids) > maxLength {
		return nil, fmt.Errorf("MOSS tokenizer: prompt/audio sequence length %d exceeds max_length=%d", len(ids), maxLength)
	}
	return ids, nil
}

func (p *Processor) Decode(ids []int) string {
	if p == nil || p.Tokenizer == nil {
		return ""
	}
	return p.Tokenizer.Decode(ids)
}
