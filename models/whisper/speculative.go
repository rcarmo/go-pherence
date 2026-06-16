package whisper

// SpeculativeDecoder implements draft-verify speculative decoding for Whisper.
// A smaller/faster model (e.g., whisper-tiny) drafts tokens, and the target model
// (e.g., whisper-small) verifies them in a single forward pass over multiple positions.
type SpeculativeDecoder struct {
	Draft  *Whisper // Small fast model (drafter)
	Target *Whisper // Large accurate model (verifier)
	K      int      // Max draft tokens per step (typically 3-5)
}

// SpeculativeResult holds the result of one speculative step.
type SpeculativeResult struct {
	Accepted []int // Tokens accepted from the draft
	Bonus    int   // One bonus token from the verifier's distribution at the rejection point
}

// Step performs one speculative decoding step:
// 1. Draft K tokens using the fast model
// 2. Verify all K+1 positions in the target model (single forward)
// 3. Accept longest matching prefix + one bonus token
func (sd *SpeculativeDecoder) Step(draftState, targetState *DecoderState) SpeculativeResult {
	cfg := sd.Target.Config
	k := sd.K
	if k <= 0 {
		k = 4
	}

	// Step 1: Draft K tokens. Checkpoint the drafter first so rejected
	// draft KV can be discarded and the next step starts from the actual
	// emitted sequence, mirroring llama.cpp/MTP verifier state hygiene.
	draftCP := checkpointDecoderState(draftState)
	draftTokens := make([]int, 0, k)
	draftLogits := make([][]float32, 0, k)

	prevTok := lastToken(draftState)
	for i := 0; i < k; i++ {
		logits := sd.Draft.Decoder.ForwardToken(prevTok, draftState)
		tok := argmax(logits)
		if tok == TokenEOT {
			break
		}
		draftTokens = append(draftTokens, tok)
		draftLogits = append(draftLogits, logits)
		prevTok = tok
	}

	if len(draftTokens) == 0 {
		// Draft immediately hit EOT — just verify one token. Restore the draft
		// checkpoint and replay the verifier input so both states keep the same
		// feed-token contract for the next step.
		logits := sd.Target.Decoder.ForwardToken(lastToken(targetState), targetState)
		tok := argmax(logits)
		restoreDecoderState(draftState, draftCP)
		sd.Draft.Decoder.ForwardToken(lastToken(draftState), draftState)
		return SpeculativeResult{Bonus: tok}
	}

	// Step 2: Run verifier over [last target token] + draft tokens. This currently
	// uses a sequential implementation under ForwardTokens, but has the right
	// G+1 verifier shape for a future fused/batched decoder.
	cp := checkpointDecoderState(targetState)
	_, verifier := sd.Target.Decoder.VerifyDraftSequential(lastToken(targetState), draftTokens, targetState)
	acceptance, err := AcceptDraftTokens(draftTokens, verifier)
	restoreDecoderState(targetState, cp)
	if err != nil {
		// Defensive fallback: emit one ordinary target token from the restored state
		// and keep the draft state aligned with the same verifier input.
		logits := sd.Target.Decoder.ForwardToken(lastToken(targetState), targetState)
		restoreDecoderState(draftState, draftCP)
		sd.Draft.Decoder.ForwardToken(lastToken(draftState), draftState)
		return SpeculativeResult{Bonus: argmax(logits)}
	}

	// The verifier pass speculatively advanced targetState through all drafts.
	// Keep target and draft KV/state faithful by replaying only the emitted
	// sequence: accepted prefix plus the verifier bonus token. This mirrors the
	// future batched verifier contract without retaining rejected draft KV.
	emitted := make([]int, 0, len(acceptance.Accepted)+1)
	emitted = append(emitted, acceptance.Accepted...)
	emitted = append(emitted, acceptance.Bonus)
	for _, tok := range emitted {
		sd.Target.Decoder.ForwardToken(tok, targetState)
	}
	restoreDecoderState(draftState, draftCP)
	for _, tok := range emitted {
		sd.Draft.Decoder.ForwardToken(tok, draftState)
	}

	_ = cfg
	return SpeculativeResult{Accepted: acceptance.Accepted, Bonus: acceptance.Bonus}
}

// SpeculativeDecode runs full speculative decoding until EOT using the default
// English transcription prompt. It is correctness scaffolding only: verifier
// calls are still sequential and therefore this is not expected to speed up
// large-v3 until a batched verifier lands.
func (sd *SpeculativeDecoder) SpeculativeDecode(encoderOutput []float32, encLen int) []int {
	return sd.SpeculativeDecodePrompt(encoderOutput, encLen, TokenEnglish, TokenTranscribe)
}

// SpeculativeDecodePrompt runs speculative decoding with an explicit Whisper
// language/task prompt. The current verifier is sequential; this exists to make
// the state/acceptance contract correct before adding a true batched verifier.
func (sd *SpeculativeDecoder) SpeculativeDecodePrompt(encoderOutput []float32, encLen int, languageToken, taskToken int) []int {
	cfg := sd.Target.Config
	if languageToken == 0 {
		languageToken = TokenEnglish
	}
	if taskToken == 0 {
		taskToken = TokenTranscribe
	}

	draftState := NewDecoderState(sd.Draft.Config, encoderOutput, encLen, sd.Draft.Decoder)
	targetState := NewDecoderState(cfg, encoderOutput, encLen, sd.Target.Decoder)

	// Feed prompt
	prompt := []int{TokenSOT, languageToken, taskToken, TokenNoTimestamps}
	for _, tok := range prompt {
		sd.Draft.Decoder.ForwardToken(tok, draftState)
		sd.Target.Decoder.ForwardToken(tok, targetState)
	}

	var tokens []int
	maxTokens := cfg.MaxDecoderLength

	for len(tokens) < maxTokens {
		result := sd.Step(draftState, targetState)

		tokens = append(tokens, result.Accepted...)
		if result.Bonus == TokenEOT {
			break
		}
		tokens = append(tokens, result.Bonus)

	}

	return tokens
}

type decoderStateCheckpoint struct {
	pos       int
	lastToken int
	selfKLen  []int
	selfVLen  []int
}

func checkpointDecoderState(state *DecoderState) decoderStateCheckpoint {
	cp := decoderStateCheckpoint{pos: state.Pos, lastToken: state.LastToken}
	cp.selfKLen = make([]int, len(state.SelfKCache))
	cp.selfVLen = make([]int, len(state.SelfVCache))
	for i := range state.SelfKCache {
		cp.selfKLen[i] = len(state.SelfKCache[i])
	}
	for i := range state.SelfVCache {
		cp.selfVLen[i] = len(state.SelfVCache[i])
	}
	return cp
}

func restoreDecoderState(state *DecoderState, cp decoderStateCheckpoint) {
	state.Pos = cp.pos
	state.LastToken = cp.lastToken
	for i, n := range cp.selfKLen {
		if i < len(state.SelfKCache) && n <= len(state.SelfKCache[i]) {
			state.SelfKCache[i] = state.SelfKCache[i][:n]
		}
	}
	for i, n := range cp.selfVLen {
		if i < len(state.SelfVCache) && n <= len(state.SelfVCache[i]) {
			state.SelfVCache[i] = state.SelfVCache[i][:n]
		}
	}
}

func lastToken(state *DecoderState) int {
	if state == nil || state.LastToken < 0 {
		return TokenNoTimestamps
	}
	return state.LastToken
}
