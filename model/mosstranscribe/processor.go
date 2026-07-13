package mosstranscribe

import "fmt"

const (
	AudioTokensPerSecond  = 12.5
	TimeMarkerEverySecond = 5
)

// AudioSpanIDs reproduces the upstream processor's placeholder expansion. Each
// marker replaces no audio features: numeric token IDs are inserted between
// runs while the number of audio placeholder IDs remains audioTokens.
func AudioSpanIDs(audioTokens, audioTokenID int, digitTokenIDs [10]int, markerEverySeconds int) []int {
	if audioTokens <= 0 {
		return nil
	}
	if markerEverySeconds <= 0 {
		out := make([]int, audioTokens)
		for i := range out {
			out[i] = audioTokenID
		}
		return out
	}
	tokensPerMarker := int(AudioTokensPerSecond * float64(markerEverySeconds))
	if tokensPerMarker <= 0 {
		return AudioSpanIDs(audioTokens, audioTokenID, digitTokenIDs, 0)
	}
	duration := float64(audioTokens) / AudioTokensPerSecond
	out := make([]int, 0, audioTokens+int(duration)/markerEverySeconds*2)
	consumed := 0
	for second := markerEverySeconds; second <= int(duration); second += markerEverySeconds {
		position := (second / markerEverySeconds) * tokensPerMarker
		for consumed < position {
			out = append(out, audioTokenID)
			consumed++
		}
		out = appendDecimalTokenIDs(out, second, digitTokenIDs)
	}
	for consumed < audioTokens {
		out = append(out, audioTokenID)
		consumed++
	}
	return out
}

func appendDecimalTokenIDs(out []int, value int, digits [10]int) []int {
	if value == 0 {
		return append(out, digits[0])
	}
	var reverse [20]int
	n := 0
	for value > 0 {
		reverse[n] = value % 10
		n++
		value /= 10
	}
	for i := n - 1; i >= 0; i-- {
		out = append(out, digits[reverse[i]])
	}
	return out
}

// InsertAudioEmbeddingsTo copies token embeddings to out and replaces every
// audio placeholder row with the next adapted audio row. Counts must match
// exactly; partial insertion is rejected before writing.
func InsertAudioEmbeddingsTo(out, tokenEmbeddings, audioEmbeddings []float32, inputIDs []int, audioTokenID, hiddenSize int) error {
	if hiddenSize <= 0 || len(tokenEmbeddings) != len(inputIDs)*hiddenSize || len(out) < len(tokenEmbeddings) || len(audioEmbeddings)%hiddenSize != 0 {
		return fmt.Errorf("MOSS insertion: malformed embeddings tokens=%d hidden=%d token_values=%d audio_values=%d out=%d", len(inputIDs), hiddenSize, len(tokenEmbeddings), len(audioEmbeddings), len(out))
	}
	placeholders := 0
	for _, id := range inputIDs {
		if id == audioTokenID {
			placeholders++
		}
	}
	audioRows := len(audioEmbeddings) / hiddenSize
	if placeholders != audioRows {
		return fmt.Errorf("MOSS insertion: %d placeholders for %d audio rows", placeholders, audioRows)
	}
	copy(out, tokenEmbeddings)
	audioRow := 0
	for token, id := range inputIDs {
		if id != audioTokenID {
			continue
		}
		copy(out[token*hiddenSize:(token+1)*hiddenSize], audioEmbeddings[audioRow*hiddenSize:(audioRow+1)*hiddenSize])
		audioRow++
	}
	return nil
}
