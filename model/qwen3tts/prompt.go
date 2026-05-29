package qwen3tts

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/rcarmo/go-pherence/loader/tokenizer"
)

// PromptIDs keeps the text-token and codec/control streams separate. Qwen3-TTS
// overlays the final text position with CodecBOS when constructing talker input
// embeddings, so callers should not collapse these streams prematurely.
type PromptIDs struct {
	Text  []uint32 `json:"text"`
	Codec []uint32 `json:"codec"`
}

// LoadTokenizer loads the Qwen tokenizer files from a model directory. Prefer
// tokenizer.json, but accept the older vocab.json + merges.txt split so local
// checkpoints can be inspected even when exported without the combined file.
func LoadTokenizer(dir string) (*tokenizer.Tokenizer, error) {
	if dir == "" {
		return nil, fmt.Errorf("empty tokenizer directory")
	}
	combined := filepath.Join(dir, "tokenizer.json")
	if _, err := os.Stat(combined); err == nil {
		return tokenizer.Load(combined)
	} else if !os.IsNotExist(err) {
		return nil, err
	}
	return loadVocabMerges(filepath.Join(dir, "vocab.json"), filepath.Join(dir, "merges.txt"))
}

// BuildCustomVoicePrompt tokenizes text and prepends the deterministic
// CustomVoice control prefix. The returned prompt contains the complete text
// stream and codec/control stream expected by the talker prefill path.
func BuildCustomVoicePrompt(tok *tokenizer.Tokenizer, text string, speaker Speaker, language Language) (PromptIDs, error) {
	if tok == nil {
		return PromptIDs{}, fmt.Errorf("nil tokenizer")
	}
	ids := tok.Encode(text)
	if len(ids) == 0 {
		return PromptIDs{}, fmt.Errorf("text produced no tokenizer IDs")
	}
	for _, id := range ids {
		if id < 0 {
			return PromptIDs{}, fmt.Errorf("negative tokenizer ID %d", id)
		}
	}
	prefixText, codec, err := CustomVoicePrefixIDs(uint32(ids[0]), speaker, language)
	if err != nil {
		return PromptIDs{}, err
	}
	textIDs := append([]uint32(nil), prefixText...)
	for _, id := range ids[1:] {
		textIDs = append(textIDs, uint32(id))
	}
	return PromptIDs{Text: textIDs, Codec: append([]uint32(nil), codec...)}, nil
}

func loadVocabMerges(vocabPath, mergesPath string) (*tokenizer.Tokenizer, error) {
	data, err := os.ReadFile(vocabPath)
	if err != nil {
		return nil, err
	}
	var vocab map[string]int
	if err := json.Unmarshal(data, &vocab); err != nil {
		return nil, err
	}
	if len(vocab) == 0 {
		return nil, fmt.Errorf("empty vocab in %s", vocabPath)
	}
	t := &tokenizer.Tokenizer{Vocab: vocab, InvVocab: make(map[int]string, len(vocab))}
	for tok, id := range vocab {
		t.InvVocab[id] = tok
	}
	mergeData, err := os.ReadFile(mergesPath)
	if err != nil {
		if os.IsNotExist(err) {
			return t, nil
		}
		return nil, err
	}
	for lineNo, line := range strings.Split(string(mergeData), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.Fields(line)
		if len(parts) != 2 {
			return nil, fmt.Errorf("malformed merge %s:%d", mergesPath, lineNo+1)
		}
		t.Merges = append(t.Merges, [2]string{parts[0], parts[1]})
	}
	return t, nil
}
