package whisper

// ForwardTokens runs a verifier block over a sequence of input tokens and
// returns one logits row per input token. The current implementation is
// intentionally sequential and preserves ForwardToken semantics; it provides the
// stable API surface needed for a future fused/batched verifier without changing
// speculative decoding callers.
func (dec *Decoder) ForwardTokens(tokens []int, state *DecoderState) [][]float32 {
	if dec == nil || state == nil || len(tokens) == 0 {
		return nil
	}
	logits := make([][]float32, len(tokens))
	for i, tok := range tokens {
		logits[i] = dec.ForwardToken(tok, state)
	}
	return logits
}

// VerifyDraftSequential verifies drafted tokens with the target decoder using
// ForwardTokens. It returns the same greedy acceptance shape used by MTP-style
// speculative decoding: verifier has G+1 rows for G drafted tokens, where the
// final row is the bonus token distribution when all drafts are accepted.
func (dec *Decoder) VerifyDraftSequential(inputToken int, drafted []int, state *DecoderState) ([][]float32, []int) {
	if dec == nil || state == nil {
		return nil, nil
	}
	inputs := make([]int, 0, len(drafted)+1)
	inputs = append(inputs, inputToken)
	inputs = append(inputs, drafted...)
	logits := dec.ForwardTokens(inputs, state)
	greedy := make([]int, len(logits))
	for i, row := range logits {
		greedy[i] = argmax(row)
	}
	return logits, greedy
}
