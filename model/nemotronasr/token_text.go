package nemotronasr

import (
	"fmt"
	"strings"

	"github.com/rcarmo/go-pherence/loader/tokenizer"
)

// DecodeRNNTText decodes raw RNN-T decisions with the released Parakeet
// vocabulary. It drops blank ID 13087, preserves repeated nonblank IDs and
// applies the Metaspace decoder's leading-space trim. The caller loads the
// matching tokenizer.json; this helper does not own tokeniser state.
func DecodeRNNTText(vocab *tokenizer.Tokenizer, decisions []int) (string, error) {
	if vocab == nil || len(vocab.InvVocab) == 0 {
		return "", fmt.Errorf("invalid Nemotron ASR tokeniser")
	}
	ids := make([]int, 0, len(decisions))
	for _, id := range decisions {
		if id == rnntBlank {
			continue
		}
		if id < 0 || id >= rnntBlank {
			return "", fmt.Errorf("invalid Nemotron ASR token ID %d", id)
		}
		value, ok := vocab.InvVocab[id]
		if !ok {
			return "", fmt.Errorf("missing Nemotron ASR token ID %d", id)
		}
		if _, special := vocab.AddedSpecial[value]; !special {
			ids = append(ids, id)
		}
	}
	return strings.TrimPrefix(vocab.Decode(ids), " "), nil
}
