package model

import (
	"fmt"
	"sync/atomic"

	"github.com/rcarmo/go-pherence/runtime/kv"
)

var gemma4SessionIDs atomic.Uint64

// Gemma4DecodeSession is the first tailored resumable inference target. It
// reuses the validated Gemma4 prompt/verifier graph so PLI, mixed head widths,
// shared K/V ownership and AVX2/SIMD dispatch stay identical to MTP execution.
// NVIDIA is reserved in the generic contract but is not accepted here until a
// stateful GPU verifier/session owns device KV directly.
type Gemma4DecodeSession struct {
	id              uint64
	model           *LlamaModel
	opts            SessionOptions
	prompt          []int
	output          []int
	state           *cpuTokenState
	generated       int
	bootstrapReplay bool
	pendingLogits   []float32
	pendingToken    int
	prefilled       bool
	closed          bool
	finished        bool
	finish          FinishReason
	stopTokens      map[int]struct{}
}

type gemma4SessionCheckpoint struct {
	sessionID     uint64
	outputLen     int
	generated     int
	finished      bool
	finish        FinishReason
	pendingLogits []float32
	pendingToken  int
	kv            kv.FloatKVCheckpoint
}

func (gemma4SessionCheckpoint) inferenceSessionCheckpoint() {}

func NewGemma4DecodeSession(m *LlamaModel, opts SessionOptions) (*Gemma4DecodeSession, error) {
	if m == nil {
		return nil, fmt.Errorf("nil Gemma4 model")
	}
	if err := validateSessionOptions(opts); err != nil {
		return nil, err
	}
	if m.Config.ModelType != "gemma4_text" {
		return nil, fmt.Errorf("model type=%q, want gemma4_text", m.Config.ModelType)
	}
	if opts.Backend == "" {
		opts.Backend = InferenceBackendSIMD
	}
	if opts.Backend != InferenceBackendSIMD {
		return nil, fmt.Errorf("Gemma4 %s decode session is not implemented", opts.Backend)
	}
	stops := make(map[int]struct{}, len(opts.StopTokenIDs))
	for _, tok := range opts.StopTokenIDs {
		if tok >= m.Config.VocabSize {
			return nil, fmt.Errorf("stop token=%d outside vocab=%d", tok, m.Config.VocabSize)
		}
		stops[tok] = struct{}{}
	}
	return &Gemma4DecodeSession{id: gemma4SessionIDs.Add(1), model: m, opts: opts, stopTokens: stops}, nil
}

// OutputTokens returns a defensive copy of the prepared prompt plus generated
// tokens. It is intended for orchestration and parity tests, not hot-path use.
func (s *Gemma4DecodeSession) OutputTokens() []int {
	if s == nil {
		return nil
	}
	return append([]int(nil), s.output...)
}

// BootstrapReplay reports whether prefill used one legacy Generate pass to
// preserve first-token parity while the session-native prefill tail is pending.
func (s *Gemma4DecodeSession) BootstrapReplay() bool {
	return s != nil && s.bootstrapReplay
}

func (s *Gemma4DecodeSession) Backend() InferenceBackend {
	if s == nil {
		return ""
	}
	return s.opts.Backend
}

// PrefillChunk currently accepts the complete prompt once. The API is chunk
// shaped deliberately; scheduler-driven partial prefill will extend this method
// after the exact single-prefill session boundary is proven.
func (s *Gemma4DecodeSession) PrefillChunk(tokens []int) (PrefillResult, error) {
	if err := s.usable(); err != nil {
		return PrefillResult{}, err
	}
	if s.prefilled {
		return PrefillResult{}, fmt.Errorf("Gemma4 session prompt is already prefilled")
	}
	if len(tokens) == 0 {
		return PrefillResult{}, fmt.Errorf("Gemma4 session prompt is empty")
	}
	for i, tok := range tokens {
		if tok < 0 || tok >= s.model.Config.VocabSize {
			return PrefillResult{}, fmt.Errorf("prompt token[%d]=%d outside vocab=%d", i, tok, s.model.Config.VocabSize)
		}
	}
	prepared := s.model.prepareGenerateTokens(tokens)
	state, err := newCPUTokenStateForLegacyGenerate(s.model, prepared, s.opts.MaxTokens)
	if err != nil {
		return PrefillResult{}, fmt.Errorf("Gemma4 session state: %w", err)
	}
	for pos, tok := range prepared {
		next, logits, emit, err := s.model.runLegacyCPUToken(state, tok, pos, nil)
		if err != nil {
			return PrefillResult{}, fmt.Errorf("Gemma4 session prefill token %d: %w", pos, err)
		}
		if pos == len(prepared)-1 {
			if !emit || len(logits) != s.model.Config.VocabSize {
				return PrefillResult{}, fmt.Errorf("Gemma4 session final prompt token did not emit logits")
			}
			s.pendingToken = next
			s.pendingLogits = append([]float32(nil), logits...)
		} else if emit {
			return PrefillResult{}, fmt.Errorf("Gemma4 session prompt token %d emitted early", pos)
		}
	}
	s.state = state
	s.prompt = append([]int(nil), prepared...)
	s.output = append([]int(nil), prepared...)
	s.state.output = s.output
	s.bootstrapReplay = false
	s.prefilled = true
	if s.opts.MaxTokens == 0 {
		s.finished, s.finish = true, FinishReasonLength
	}
	return PrefillResult{ConsumedTokens: len(prepared), Position: len(prepared), ReadyToDecode: !s.finished}, nil
}

func (s *Gemma4DecodeSession) DecodeStep() (DecodeResult, error) {
	if err := s.usable(); err != nil {
		return DecodeResult{}, err
	}
	if !s.prefilled {
		return DecodeResult{}, fmt.Errorf("Gemma4 session prompt is not prefilled")
	}
	if s.finished {
		return DecodeResult{Position: len(s.output), Generated: s.generated, Finished: true, FinishReason: s.finish}, nil
	}
	var tok int
	var logits []float32
	if s.pendingLogits != nil {
		tok = s.pendingToken
		logits = s.pendingLogits
		s.pendingLogits = nil
		s.pendingToken = 0
	} else {
		input := s.output[len(s.output)-1]
		pos := len(s.output) - 1
		var emit bool
		var err error
		tok, logits, emit, err = s.model.runLegacyCPUToken(s.state, input, pos, nil)
		if err != nil {
			return DecodeResult{}, fmt.Errorf("Gemma4 session decode position %d: %w", pos, err)
		}
		if !emit {
			return DecodeResult{}, fmt.Errorf("Gemma4 session decode position %d did not emit", pos)
		}
	}
	s.output = append(s.output, tok)
	s.state.output = s.output
	s.generated++
	if _, stop := s.stopTokens[tok]; stop {
		s.finished, s.finish = true, FinishReasonStopToken
	} else if s.generated >= s.opts.MaxTokens {
		s.finished, s.finish = true, FinishReasonLength
	}
	return DecodeResult{Token: tok, Logits: append([]float32(nil), logits...), Position: len(s.output), Generated: s.generated, Finished: s.finished, FinishReason: s.finish}, nil
}

func (s *Gemma4DecodeSession) Checkpoint() (SessionCheckpoint, error) {
	if err := s.usable(); err != nil {
		return nil, err
	}
	if !s.prefilled {
		return nil, fmt.Errorf("Gemma4 session prompt is not prefilled")
	}
	return gemma4SessionCheckpoint{sessionID: s.id, outputLen: len(s.output), generated: s.generated, finished: s.finished, finish: s.finish, pendingLogits: append([]float32(nil), s.pendingLogits...), pendingToken: s.pendingToken, kv: kv.CheckpointFloatKV(s.state.kvCacheK, s.state.kvCacheV)}, nil
}

func (s *Gemma4DecodeSession) Restore(checkpoint SessionCheckpoint) error {
	if err := s.usable(); err != nil {
		return err
	}
	cp, ok := checkpoint.(gemma4SessionCheckpoint)
	if !ok || cp.sessionID != s.id {
		return fmt.Errorf("checkpoint does not belong to this Gemma4 session")
	}
	if cp.outputLen < len(s.prompt) || cp.outputLen > len(s.output) || cp.generated != cp.outputLen-len(s.prompt) {
		return fmt.Errorf("invalid Gemma4 session checkpoint output=%d generated=%d", cp.outputLen, cp.generated)
	}
	if err := cp.kv.Restore(s.state.kvCacheK, s.state.kvCacheV); err != nil {
		return err
	}
	s.output = s.output[:cp.outputLen]
	s.state.output = s.output
	s.generated, s.finished, s.finish = cp.generated, cp.finished, cp.finish
	s.pendingLogits = append(s.pendingLogits[:0], cp.pendingLogits...)
	s.pendingToken = cp.pendingToken
	return nil
}

func (s *Gemma4DecodeSession) Finished() (bool, FinishReason) {
	if s == nil {
		return true, FinishReasonClosed
	}
	return s.finished || s.closed, func() FinishReason {
		if s.closed {
			return FinishReasonClosed
		}
		return s.finish
	}()
}

func (s *Gemma4DecodeSession) Close() error {
	if s == nil || s.closed {
		return nil
	}
	s.closed, s.finished, s.finish = true, true, FinishReasonClosed
	s.prompt, s.output, s.pendingLogits, s.state = nil, nil, nil, nil
	return nil
}

func (s *Gemma4DecodeSession) usable() error {
	if s == nil {
		return fmt.Errorf("nil Gemma4 session")
	}
	if s.closed {
		return fmt.Errorf("Gemma4 session is closed")
	}
	return nil
}
