package model

import "fmt"

// Gemma4ExperimentalDecodeBatchMetadata is the flattened per-request metadata
// for one experimental static decode step. It deliberately keeps request state
// ownership on each session while exposing the token/position/length vectors a
// future wider lowering would consume directly.
type Gemma4ExperimentalDecodeBatchMetadata struct {
	Batch             int
	TokensFlat        []int
	PositionsFlat     []int
	OutputLengthsFlat []int
}

// Gemma4ExperimentalDecodeBatchResult reports one experimental multi-session
// decode step. UsedBatchFinish is true when the step completed its final
// norm/LM-head through FinishCPUDecodeBatch; otherwise the step fell back to
// the ordinary per-session DecodeStep path and FallbackReason explains why.
type Gemma4ExperimentalDecodeBatchResult struct {
	Metadata        Gemma4ExperimentalDecodeBatchMetadata
	Results         []DecodeResult
	UsedBatchFinish bool
	FallbackReason  string
}

// RunGemma4ExperimentalDecodeBatch runs one non-default static Gemma4 decode
// step over 1/2/4/8 independent sessions. The transformer body remains
// request-local so each session preserves its own KV/state; only the final
// norm/LM-head tail is batched when it is safe to do so.
func RunGemma4ExperimentalDecodeBatch(sessions []*Gemma4DecodeSession) (Gemma4ExperimentalDecodeBatchResult, error) {
	meta, model, err := newGemma4ExperimentalDecodeBatchMetadata(sessions)
	if err != nil {
		return Gemma4ExperimentalDecodeBatchResult{}, err
	}
	if gemma4ExperimentalDecodeBatchNeedsSequentialFallback(sessions) {
		return runGemma4ExperimentalDecodeBatchSequential(sessions, meta, "pending_prefill_boundary_logits")
	}
	results, err := runGemma4ExperimentalDecodeBatchTail(model, sessions, meta)
	if err != nil {
		return Gemma4ExperimentalDecodeBatchResult{}, err
	}
	return results, nil
}

func newGemma4ExperimentalDecodeBatchMetadata(sessions []*Gemma4DecodeSession) (Gemma4ExperimentalDecodeBatchMetadata, *LlamaModel, error) {
	B := len(sessions)
	if !gemma4ExperimentalDecodeBatchSizeSupported(B) {
		return Gemma4ExperimentalDecodeBatchMetadata{}, nil, fmt.Errorf("Gemma4 experimental decode batch size=%d unsupported, want 1/2/4/8", B)
	}
	meta := Gemma4ExperimentalDecodeBatchMetadata{
		Batch:             B,
		TokensFlat:        make([]int, B),
		PositionsFlat:     make([]int, B),
		OutputLengthsFlat: make([]int, B),
	}
	var model *LlamaModel
	for i, s := range sessions {
		if err := validateGemma4ExperimentalDecodeBatchSession(s); err != nil {
			return Gemma4ExperimentalDecodeBatchMetadata{}, nil, fmt.Errorf("Gemma4 experimental decode batch session %d: %w", i, err)
		}
		if i == 0 {
			model = s.model
		} else if s.model != model {
			return Gemma4ExperimentalDecodeBatchMetadata{}, nil, fmt.Errorf("Gemma4 experimental decode batch session %d model does not match session 0", i)
		}
		meta.OutputLengthsFlat[i] = len(s.output)
		meta.PositionsFlat[i] = len(s.output) - 1
		meta.TokensFlat[i] = s.output[len(s.output)-1]
	}
	return meta, model, nil
}

func validateGemma4ExperimentalDecodeBatchSession(s *Gemma4DecodeSession) error {
	if err := s.usable(); err != nil {
		return err
	}
	if s.model == nil {
		return fmt.Errorf("nil Gemma4 model")
	}
	if s.model.Config.ModelType != "gemma4_text" {
		return fmt.Errorf("model type=%q, want gemma4_text", s.model.Config.ModelType)
	}
	if s.opts.Backend != InferenceBackendSIMD {
		return fmt.Errorf("backend=%q, want %q", s.opts.Backend, InferenceBackendSIMD)
	}
	if !s.prefilled {
		if s.prefillStarted {
			return fmt.Errorf("Gemma4 session prompt prefill is incomplete")
		}
		return fmt.Errorf("Gemma4 session prompt is not prefilled")
	}
	if s.finished {
		return fmt.Errorf("Gemma4 session is finished")
	}
	if s.state == nil {
		return fmt.Errorf("Gemma4 session state is nil")
	}
	if len(s.output) == 0 {
		return fmt.Errorf("Gemma4 session output is empty")
	}
	if len(s.state.output) != len(s.output) {
		return fmt.Errorf("Gemma4 session state output len=%d, want %d", len(s.state.output), len(s.output))
	}
	if s.pendingLogits != nil && len(s.pendingLogits) != s.model.Config.VocabSize {
		return fmt.Errorf("Gemma4 session pending logits len=%d, want vocab=%d", len(s.pendingLogits), s.model.Config.VocabSize)
	}
	return nil
}

func gemma4ExperimentalDecodeBatchSizeSupported(n int) bool {
	switch n {
	case 1, 2, 4, 8:
		return true
	default:
		return false
	}
}

func gemma4ExperimentalDecodeBatchNeedsSequentialFallback(sessions []*Gemma4DecodeSession) bool {
	for _, s := range sessions {
		if s != nil && s.pendingLogits != nil {
			return true
		}
	}
	return false
}

func runGemma4ExperimentalDecodeBatchSequential(sessions []*Gemma4DecodeSession, meta Gemma4ExperimentalDecodeBatchMetadata, reason string) (Gemma4ExperimentalDecodeBatchResult, error) {
	out := Gemma4ExperimentalDecodeBatchResult{Metadata: meta, Results: make([]DecodeResult, len(sessions)), FallbackReason: reason}
	for i, s := range sessions {
		res, err := s.DecodeStep()
		if err != nil {
			return Gemma4ExperimentalDecodeBatchResult{}, fmt.Errorf("Gemma4 experimental decode batch session %d fallback step: %w", i, err)
		}
		out.Results[i] = res
	}
	return out, nil
}

func runGemma4ExperimentalDecodeBatchTail(model *LlamaModel, sessions []*Gemma4DecodeSession, meta Gemma4ExperimentalDecodeBatchMetadata) (Gemma4ExperimentalDecodeBatchResult, error) {
	B, h := meta.Batch, model.Config.HiddenSize
	hiddenFlat, ok := checkedProduct(B, h)
	if !ok {
		return Gemma4ExperimentalDecodeBatchResult{}, fmt.Errorf("Gemma4 experimental decode batch hidden size overflow B=%d hidden=%d", B, h)
	}
	flatHidden := make([]float32, hiddenFlat)
	hiddenRows := make([][]float32, B)
	for i := range hiddenRows {
		hiddenRows[i] = flatHidden[i*h : (i+1)*h]
	}
	for i, s := range sessions {
		state := s.state
		if state.captureFinalHidden != nil || state.skipFinalDecode {
			return Gemma4ExperimentalDecodeBatchResult{}, fmt.Errorf("Gemma4 experimental decode batch session %d already has a deferred final decode", i)
		}
		captured := false
		state.captureFinalHidden = func(pos int, hidden []float32) {
			copy(hiddenRows[i], hidden)
			captured = true
		}
		state.skipFinalDecode = true
		_, _, emit, err := model.runLegacyCPUToken(state, meta.TokensFlat[i], meta.PositionsFlat[i], nil)
		state.captureFinalHidden = nil
		state.skipFinalDecode = false
		if err != nil {
			return Gemma4ExperimentalDecodeBatchResult{}, fmt.Errorf("Gemma4 experimental decode batch session %d token %d at position %d: %w", i, meta.TokensFlat[i], meta.PositionsFlat[i], err)
		}
		if !emit {
			return Gemma4ExperimentalDecodeBatchResult{}, fmt.Errorf("Gemma4 experimental decode batch session %d token %d at position %d did not emit", i, meta.TokensFlat[i], meta.PositionsFlat[i])
		}
		if !captured {
			return Gemma4ExperimentalDecodeBatchResult{}, fmt.Errorf("Gemma4 experimental decode batch session %d token %d at position %d did not capture final hidden", i, meta.TokensFlat[i], meta.PositionsFlat[i])
		}
	}
	finalActivations, logitsRows, tokens, err := model.FinishCPUDecodeBatch(hiddenRows)
	usedBatchFinish := true
	fallbackReason := ""
	if err != nil {
		usedBatchFinish = false
		fallbackReason = "sequential_final_finish"
		finalActivations = make([][]float32, B)
		logitsRows = make([][]float32, B)
		tokens = make([]int, B)
		for i := range hiddenRows {
			act, logits, tok, ferr := model.FinishCPUDecodeStep(hiddenRows[i])
			if ferr != nil {
				return Gemma4ExperimentalDecodeBatchResult{}, fmt.Errorf("Gemma4 experimental decode batch final finish row %d after batch error %v: %w", i, err, ferr)
			}
			finalActivations[i], logitsRows[i], tokens[i] = act, logits, tok
		}
	}
	_ = finalActivations
	out := Gemma4ExperimentalDecodeBatchResult{
		Metadata:        meta,
		Results:         make([]DecodeResult, B),
		UsedBatchFinish: usedBatchFinish,
		FallbackReason:  fallbackReason,
	}
	for i, s := range sessions {
		out.Results[i] = s.commitDecodeStep(tokens[i], logitsRows[i])
	}
	return out, nil
}
