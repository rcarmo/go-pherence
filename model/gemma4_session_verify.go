package model

import "fmt"

// Verify greedily verifies a variable-length draft against this session's
// ordinary decode path. It stages verifier tokens behind a checkpoint, restores
// the request, then replays only the accepted prefix plus bonus token so output,
// KV ownership, stop state and subsequent DecodeStep behaviour stay identical
// to non-speculative generation.
//
// This initial session bridge is deliberately correctness-first and recomputes
// committed tokens during replay. Batched verifier lowering can replace the
// staging loop without changing the request-facing contract.
func (s *Gemma4DecodeSession) Verify(drafted []int) (acceptance MTPAcceptance, err error) {
	if err := s.usable(); err != nil {
		return MTPAcceptance{}, err
	}
	if !s.prefilled {
		return MTPAcceptance{}, fmt.Errorf("Gemma4 session prompt is not prefilled")
	}
	if s.finished {
		return MTPAcceptance{}, fmt.Errorf("Gemma4 session is already finished: %s", s.finish)
	}
	if len(drafted) == 0 || len(drafted) > maxMTPDraftCount {
		return MTPAcceptance{}, fmt.Errorf("Gemma4 verifier draft count=%d outside [1,%d]", len(drafted), maxMTPDraftCount)
	}
	for i, tok := range drafted {
		if tok < 0 || tok >= s.model.Config.VocabSize {
			return MTPAcceptance{}, fmt.Errorf("Gemma4 verifier draft token[%d]=%d outside vocab=%d", i, tok, s.model.Config.VocabSize)
		}
	}
	verifyN := len(drafted) + 1
	if s.opts.MaxTokens > 0 && s.generated+verifyN > s.opts.MaxTokens {
		return MTPAcceptance{}, fmt.Errorf("Gemma4 verifier needs %d tokens with %d/%d already generated", verifyN, s.generated, s.opts.MaxTokens)
	}

	checkpoint, err := s.Checkpoint()
	if err != nil {
		return MTPAcceptance{}, err
	}
	restored := false
	defer func() {
		if err != nil && !restored {
			if restoreErr := s.Restore(checkpoint); restoreErr != nil {
				err = fmt.Errorf("%v; restore verifier checkpoint: %w", err, restoreErr)
			}
		}
	}()

	verifierTokens := make([]int, 0, verifyN)
	for i := 0; i < verifyN; i++ {
		step, stepErr := s.DecodeStep()
		if stepErr != nil {
			return MTPAcceptance{}, fmt.Errorf("Gemma4 verifier token %d: %w", i, stepErr)
		}
		verifierTokens = append(verifierTokens, step.Token)
		if step.Finished && i+1 < verifyN {
			return MTPAcceptance{}, fmt.Errorf("Gemma4 verifier reached %s after token %d", step.FinishReason, i)
		}
	}
	acceptance, err = AcceptMTPDraft(drafted, verifierTokens)
	if err != nil {
		return MTPAcceptance{}, err
	}
	if err = s.Restore(checkpoint); err != nil {
		return MTPAcceptance{}, fmt.Errorf("restore Gemma4 verifier checkpoint: %w", err)
	}

	for i, want := range acceptance.OutputTokens {
		step, stepErr := s.DecodeStep()
		if stepErr != nil {
			return MTPAcceptance{}, fmt.Errorf("replay accepted verifier token %d: %w", i, stepErr)
		}
		if step.Token != want {
			return MTPAcceptance{}, fmt.Errorf("replay accepted verifier token %d=%d, want %d", i, step.Token, want)
		}
	}
	restored = true
	return acceptance, nil
}
