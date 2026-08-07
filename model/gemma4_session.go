package model

import (
	"errors"
	"fmt"
	"sync/atomic"

	"github.com/rcarmo/go-pherence/runtime/kv"
	"github.com/rcarmo/go-pherence/runtime/promptcache"
)

var gemma4SessionIDs atomic.Uint64

// Gemma4DecodeSession is the first tailored resumable inference target. It
// reuses the validated Gemma4 prompt/verifier graph so PLI, mixed head widths,
// shared K/V ownership and AVX2/SIMD dispatch stay identical to MTP execution.
// NVIDIA is reserved in the generic contract but is not accepted here until a
// stateful GPU verifier/session owns device KV directly.
type Gemma4DecodeSession struct {
	id                   uint64
	model                *LlamaModel
	opts                 SessionOptions
	prompt               []int
	output               []int
	state                *cpuTokenState
	prefillPrepared      []int
	prefillCursor        int
	generated            int
	bootstrapReplay      bool
	pendingLogits        []float32
	pendingToken         int
	prefillStarted       bool
	prefilled            bool
	closed               bool
	finished             bool
	finish               FinishReason
	stopTokens           map[int]struct{}
	promptCache          *promptcache.Cache
	promptCacheIdentity  promptcache.Identity
	promptCacheBlockSize int
}

type Gemma4PromptCacheConfig struct {
	Cache     *promptcache.Cache
	Identity  promptcache.Identity
	BlockSize int
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
	return NewGemma4DecodeSessionWithPromptCache(m, opts, Gemma4PromptCacheConfig{})
}

func NewGemma4DecodeSessionWithPromptCache(m *LlamaModel, opts SessionOptions, cacheCfg Gemma4PromptCacheConfig) (*Gemma4DecodeSession, error) {
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
	if err := cacheCfg.validate(); err != nil {
		return nil, err
	}
	stops := make(map[int]struct{}, len(opts.StopTokenIDs))
	for _, tok := range opts.StopTokenIDs {
		if tok >= m.Config.VocabSize {
			return nil, fmt.Errorf("stop token=%d outside vocab=%d", tok, m.Config.VocabSize)
		}
		stops[tok] = struct{}{}
	}
	return &Gemma4DecodeSession{
		id:                   gemma4SessionIDs.Add(1),
		model:                m,
		opts:                 opts,
		stopTokens:           stops,
		promptCache:          cacheCfg.Cache,
		promptCacheIdentity:  cacheCfg.Identity,
		promptCacheBlockSize: cacheCfg.BlockSize,
	}, nil
}

func (cfg Gemma4PromptCacheConfig) enabled() bool {
	return cfg.Cache != nil || cfg.BlockSize != 0 || cfg.Identity != (promptcache.Identity{})
}

func (cfg Gemma4PromptCacheConfig) validate() error {
	if !cfg.enabled() {
		return nil
	}
	if cfg.Cache == nil {
		return fmt.Errorf("Gemma4 prompt cache config requires cache")
	}
	if cfg.BlockSize <= 0 {
		return promptcache.ErrInvalidBlockSize
	}
	return cfg.Identity.Validate()
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

func (s *Gemma4DecodeSession) BeginPrefill(tokens []int) error {
	if err := s.usable(); err != nil {
		return err
	}
	if s.prefillStarted || s.prefilled {
		return fmt.Errorf("Gemma4 session prompt prefill already started")
	}
	if len(tokens) == 0 {
		return fmt.Errorf("Gemma4 session prompt is empty")
	}
	for i, tok := range tokens {
		if tok < 0 || tok >= s.model.Config.VocabSize {
			return fmt.Errorf("prompt token[%d]=%d outside vocab=%d", i, tok, s.model.Config.VocabSize)
		}
	}
	prepared := s.model.prepareGenerateTokens(tokens)
	stateMaxTokens := s.opts.MaxTokens
	if stateMaxTokens == 0 {
		stateMaxTokens = 1
	}
	state, err := newCPUTokenStateForLegacyGenerate(s.model, prepared, stateMaxTokens)
	if err != nil {
		return fmt.Errorf("Gemma4 session state: %w", err)
	}
	if s.promptCache != nil && state.compressedKV != nil {
		return fmt.Errorf("Gemma4 session prompt cache requires float KV")
	}
	startPos := 0
	var cachedBoundaryLogits []float32
	var cachedBoundaryToken int
	if s.promptCache != nil {
		snap, ok, err := s.promptCache.FindLongest(s.promptCacheIdentity, s.promptCacheBlockSize, prepared)
		if err != nil {
			return fmt.Errorf("Gemma4 session prompt cache lookup: %w", err)
		}
		if ok {
			cached, ok := snap.(*Gemma4PromptSnapshot)
			if !ok {
				return fmt.Errorf("Gemma4 session prompt cache snapshot %T", snap)
			}
			if err := cached.restoreInto(s.model, state); err != nil {
				return fmt.Errorf("Gemma4 session prompt cache restore: %w", err)
			}
			startPos = cached.Position()
			cachedBoundaryLogits = append([]float32(nil), cached.BoundaryLogits...)
			cachedBoundaryToken = cached.BoundaryToken
		}
	}
	state.output = state.output[:len(prepared)]
	state.position = startPos
	s.state = state
	s.prefillPrepared = append([]int(nil), prepared...)
	s.prefillCursor = startPos
	s.generated = 0
	s.bootstrapReplay = false
	s.pendingLogits = nil
	s.pendingToken = 0
	s.prefillStarted = true
	s.prefilled = false
	s.finished = false
	s.finish = FinishReasonNone
	if startPos == len(prepared) {
		if err := s.completePrefill(cachedBoundaryLogits, cachedBoundaryToken); err != nil {
			s.state = nil
			s.prefillPrepared = nil
			s.prefillCursor = 0
			s.prefillStarted = false
			return err
		}
	}
	return nil
}

func (s *Gemma4DecodeSession) RemainingPrefill() int {
	if s == nil || !s.prefillStarted || s.prefilled {
		return 0
	}
	if rem := len(s.prefillPrepared) - s.prefillCursor; rem > 0 {
		return rem
	}
	return 0
}

func (s *Gemma4DecodeSession) PrefillNext(limit int) (PrefillResult, error) {
	if err := s.usable(); err != nil {
		return PrefillResult{}, err
	}
	if !s.prefillStarted {
		return PrefillResult{}, fmt.Errorf("Gemma4 session prompt prefill is not started")
	}
	if s.prefilled {
		return PrefillResult{}, fmt.Errorf("Gemma4 session prompt is already prefilled")
	}
	if limit <= 0 {
		return PrefillResult{}, fmt.Errorf("Gemma4 session prefill limit=%d must be positive", limit)
	}
	remaining := len(s.prefillPrepared) - s.prefillCursor
	if remaining <= 0 {
		return PrefillResult{}, fmt.Errorf("Gemma4 session prompt prefill has no remaining tokens")
	}
	if limit > remaining {
		limit = remaining
	}
	start := s.prefillCursor
	end := start + limit
	var boundaryLogits []float32
	var boundaryToken int
	for pos := start; pos < end; pos++ {
		next, logits, emit, err := s.model.runLegacyCPUToken(s.state, s.prefillPrepared[pos], pos, nil)
		if err != nil {
			return PrefillResult{}, fmt.Errorf("Gemma4 session prefill token %d: %w", pos, err)
		}
		if emit {
			if pos != len(s.prefillPrepared)-1 {
				return PrefillResult{}, fmt.Errorf("Gemma4 session prompt token %d emitted early", pos)
			}
			boundaryToken = next
			boundaryLogits = append([]float32(nil), logits...)
		}
		s.state.position = pos + 1
		if s.promptCache != nil && (pos+1)%s.promptCacheBlockSize == 0 {
			var snapLogits []float32
			var snapToken int
			if emit {
				snapLogits, snapToken = boundaryLogits, boundaryToken
			}
			snap, err := newGemma4PromptSnapshotFromState(s.model, s.state, pos+1, snapLogits, snapToken)
			if err != nil {
				return PrefillResult{}, fmt.Errorf("Gemma4 session snapshot at %d: %w", pos+1, err)
			}
			if err := s.promptCache.Put(s.promptCacheIdentity, s.promptCacheBlockSize, s.prefillPrepared[:pos+1], snap); err != nil && !errors.Is(err, promptcache.ErrOverBudget) {
				return PrefillResult{}, fmt.Errorf("Gemma4 session prompt cache store at %d: %w", pos+1, err)
			}
		}
	}
	s.prefillCursor = end
	if s.prefillCursor == len(s.prefillPrepared) {
		if err := s.completePrefill(boundaryLogits, boundaryToken); err != nil {
			return PrefillResult{}, err
		}
	}
	position := s.prefillCursor
	if s.prefilled {
		position = len(s.prompt)
	}
	return PrefillResult{ConsumedTokens: end - start, Position: position, ReadyToDecode: s.prefilled && !s.finished}, nil
}

func (s *Gemma4DecodeSession) completePrefill(boundaryLogits []float32, boundaryToken int) error {
	if s.opts.MaxTokens > 0 {
		if len(boundaryLogits) != s.model.Config.VocabSize {
			return fmt.Errorf("Gemma4 session final prompt boundary logits unavailable after prefix restore")
		}
		s.pendingToken = boundaryToken
		s.pendingLogits = append([]float32(nil), boundaryLogits...)
	} else {
		s.pendingToken = 0
		s.pendingLogits = nil
	}
	s.prompt = append([]int(nil), s.prefillPrepared...)
	s.output = append([]int(nil), s.prefillPrepared...)
	s.state.output = s.output
	s.state.position = len(s.prompt)
	s.prefillPrepared = nil
	s.prefillCursor = 0
	s.prefilled = true
	if s.opts.MaxTokens == 0 {
		s.finished, s.finish = true, FinishReasonLength
	}
	return nil
}

// PrefillChunk preserves the original one-shot behavior by beginning prefill
// once and then draining all remaining prepared prompt tokens.
func (s *Gemma4DecodeSession) PrefillChunk(tokens []int) (PrefillResult, error) {
	if err := s.BeginPrefill(tokens); err != nil {
		return PrefillResult{}, err
	}
	if !s.prefilled {
		if _, err := s.PrefillNext(s.RemainingPrefill()); err != nil {
			return PrefillResult{}, err
		}
	}
	return PrefillResult{ConsumedTokens: len(s.prompt), Position: len(s.prompt), ReadyToDecode: !s.finished}, nil
}

func (s *Gemma4DecodeSession) DecodeStep() (DecodeResult, error) {
	if err := s.usable(); err != nil {
		return DecodeResult{}, err
	}
	if !s.prefilled {
		if s.prefillStarted {
			return DecodeResult{}, fmt.Errorf("Gemma4 session prompt prefill is incomplete")
		}
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
	return s.commitDecodeStep(tok, logits), nil
}

func (s *Gemma4DecodeSession) commitDecodeStep(tok int, logits []float32) DecodeResult {
	s.output = append(s.output, tok)
	s.state.output = s.output
	s.generated++
	if _, stop := s.stopTokens[tok]; stop {
		s.finished, s.finish = true, FinishReasonStopToken
	} else if s.generated >= s.opts.MaxTokens {
		s.finished, s.finish = true, FinishReasonLength
	}
	return DecodeResult{Token: tok, Logits: append([]float32(nil), logits...), Position: len(s.output), Generated: s.generated, Finished: s.finished, FinishReason: s.finish}
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
	s.prompt, s.output, s.pendingLogits, s.state, s.prefillPrepared = nil, nil, nil, nil, nil
	s.prefillCursor, s.prefillStarted = 0, false
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
