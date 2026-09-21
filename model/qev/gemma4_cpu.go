package qev

import (
	"context"
	"fmt"
	"sync"

	"github.com/rcarmo/go-pherence/model"
)

// Gemma4CPUScorer is the correctness oracle for QEV branch execution. It owns
// one request-local session per context, checkpoints the prefilled trunk and
// restores it before every branch. A fused implementation must match these
// candidate logits before it can replace this path.
type Gemma4CPUScorer struct {
	Model       *model.LlamaModel
	Backend     model.InferenceBackend
	PromptCache model.Gemma4PromptCacheConfig
}

// Gemma4SIMDBatchScorer is the fused sibling-path implementation. It shares a
// read-only prompt trunk while each branch owns separate suffix KV. The scalar
// checkpoint/restore Gemma4CPUScorer remains the acceptance oracle.
type Gemma4SIMDBatchScorer struct {
	Model       *model.LlamaModel
	Backend     model.InferenceBackend
	PromptCache model.Gemma4PromptCacheConfig
}

// Gemma4NVIDIAScorer uses the SIMD session for canonical prompt prefill, then
// uploads the immutable trunk into a request-owned device KV arena and executes
// every sibling transformer/LM-head projection through the resident PTX graph.
type Gemma4NVIDIAScorer struct {
	Model        *model.LlamaModel
	GPU          *model.Gemma4NVIDIA
	PromptCache  model.Gemma4PromptCacheConfig
	mu           sync.Mutex
	cachedTokens []int
	cached       *model.Gemma4NVIDIAContext
}

func (s *Gemma4NVIDIAScorer) Close() {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.cached != nil {
		s.cached.Close()
		s.cached = nil
	}
	s.cachedTokens = nil
}

func (s *Gemma4CPUScorer) ScoreContext(ctx context.Context, prompt []int, branches []Branch, allowCache bool) ([][]float32, error) {
	if s == nil {
		return nil, fmt.Errorf("nil Gemma4 CPU scorer")
	}
	session, err := prepareGemma4ScoringSession(ctx, s.Model, s.Backend, s.PromptCache, prompt, branches, allowCache)
	if err != nil {
		return nil, err
	}
	defer session.Close()
	trunk, err := session.Checkpoint()
	if err != nil {
		return nil, err
	}
	out := make([][]float32, len(branches))
	for i, branch := range branches {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if err := session.Restore(trunk); err != nil {
			return nil, fmt.Errorf("branch %d restore: %w", i, err)
		}
		var logits []float32
		for _, token := range branch.Tokens {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			step, err := session.AppendToken(token)
			if err != nil {
				return nil, fmt.Errorf("branch %d: %w", i, err)
			}
			logits = step.Logits
		}
		out[i] = make([]float32, len(branch.CandidateTokens))
		for j, token := range branch.CandidateTokens {
			out[i][j] = logits[token]
		}
	}
	return out, nil
}

func (s *Gemma4SIMDBatchScorer) ScoreContext(ctx context.Context, prompt []int, branches []Branch, allowCache bool) ([][]float32, error) {
	if s == nil {
		return nil, fmt.Errorf("nil Gemma4 SIMD batch scorer")
	}
	session, err := prepareGemma4ScoringSession(ctx, s.Model, s.Backend, s.PromptCache, prompt, branches, allowCache)
	if err != nil {
		return nil, err
	}
	defer session.Close()
	out := make([][]float32, len(branches))
	for start := 0; start < len(branches); start += DecisionSequences {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		end := min(start+DecisionSequences, len(branches))
		paths := make([][]int, end-start)
		for i := start; i < end; i++ {
			paths[i-start] = branches[i].Tokens
		}
		batch, err := session.ScoreIndependentBranches(ctx, paths)
		if err != nil {
			return nil, fmt.Errorf("branches [%d,%d): %w", start, end, err)
		}
		for i := start; i < end; i++ {
			branch := branches[i]
			out[i] = make([]float32, len(branch.CandidateTokens))
			for j, token := range branch.CandidateTokens {
				out[i][j] = batch.Logits[i-start][token]
			}
		}
	}
	return out, nil
}

func (s *Gemma4NVIDIAScorer) ScoreSplitContext(ctx context.Context, shared, contextTokens []int, branches []Branch, allowCache bool) ([][]float32, error) {
	if s == nil || s.GPU == nil || s.Model == nil {
		return nil, fmt.Errorf("nil Gemma4 NVIDIA scorer")
	}
	prompt := append(append([]int(nil), shared...), contextTokens...)
	maxDepth := 0
	if _, err := validateGemma4ScoringInput(ctx, s.Model, prompt, branches, &maxDepth); err != nil {
		return nil, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	var prefix *model.Gemma4NVIDIAContext
	hit := allowCache && sameTokenSlice(s.cachedTokens, shared) && s.cached != nil
	if hit {
		prefix = s.cached
	} else {
		if s.cached != nil {
			s.cached.Close()
			s.cached = nil
		}
		trunkCap := len(shared) + len(contextTokens) + maxDepth
		p, err := s.GPU.PrefillPreparedCapacity(ctx, shared, trunkCap, DecisionSequences, maxDepth)
		if err != nil {
			return nil, fmt.Errorf("NVIDIA shared prefill: %w", err)
		}
		if allowCache {
			s.cached = p
			s.cachedTokens = append([]int(nil), shared...)
			prefix = s.cached
		} else {
			prefix = p
			defer prefix.Close()
		}
	}
	gpuCtx, err := s.GPU.RefillPrefixed(ctx, prefix, contextTokens)
	if err != nil {
		return nil, fmt.Errorf("NVIDIA context prefill: %w", err)
	}
	defer gpuCtx.Close()
	out := make([][]float32, len(branches))
	for start := 0; start < len(branches); start += DecisionSequences {
		end := min(start+DecisionSequences, len(branches))
		paths := make([][]int, end-start)
		for i := start; i < end; i++ {
			paths[i-start] = branches[i].Tokens
		}
		batch, err := gpuCtx.ScoreIndependentBranches(ctx, paths)
		if err != nil {
			return nil, err
		}
		for i := start; i < end; i++ {
			out[i] = make([]float32, len(branches[i].CandidateTokens))
			for j, tok := range branches[i].CandidateTokens {
				out[i][j] = batch.Logits[i-start][tok]
			}
		}
	}
	return out, nil
}

func sameTokenSlice(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func (s *Gemma4NVIDIAScorer) ScoreContext(ctx context.Context, prompt []int, branches []Branch, allowCache bool) ([][]float32, error) {
	if s == nil || s.GPU == nil || s.Model == nil {
		return nil, fmt.Errorf("nil Gemma4 NVIDIA scorer")
	}
	maxDepth := 0
	if _, err := validateGemma4ScoringInput(ctx, s.Model, prompt, branches, &maxDepth); err != nil {
		return nil, err
	}
	if s.GPU.DeviceName() == "" {
		return nil, fmt.Errorf("NVIDIA scorer has no active device")
	}
	out := make([][]float32, len(branches))
	gpuCtx, err := s.GPU.PrefillPrepared(ctx, prompt, min(DecisionSequences, len(branches)), maxDepth)
	if err != nil {
		return nil, fmt.Errorf("NVIDIA prefill: %w", err)
	}
	defer gpuCtx.Close()
	for start := 0; start < len(branches); start += DecisionSequences {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		end := min(start+DecisionSequences, len(branches))
		paths := make([][]int, end-start)
		for i := start; i < end; i++ {
			paths[i-start] = branches[i].Tokens
		}
		batch, scoreErr := gpuCtx.ScoreIndependentBranches(ctx, paths)
		if scoreErr != nil {
			return nil, fmt.Errorf("NVIDIA branches [%d,%d): %w", start, end, scoreErr)
		}
		for i := start; i < end; i++ {
			out[i] = make([]float32, len(branches[i].CandidateTokens))
			for j, tok := range branches[i].CandidateTokens {
				out[i][j] = batch.Logits[i-start][tok]
			}
		}
	}
	return out, nil
}

func validateGemma4ScoringInput(ctx context.Context, m *model.LlamaModel, prompt []int, branches []Branch, maxBranchTokens *int) (int, error) {
	if ctx == nil {
		return 0, fmt.Errorf("nil context")
	}
	if m == nil {
		return 0, fmt.Errorf("nil Gemma4 model")
	}
	if m.Config.ModelType != "gemma4_text" {
		return 0, fmt.Errorf("model type=%q, want gemma4_text", m.Config.ModelType)
	}
	if len(prompt) == 0 {
		return 0, fmt.Errorf("empty prepared prompt")
	}
	if len(branches) == 0 {
		return 0, fmt.Errorf("empty branch batch")
	}
	maxDepth := 0
	for i, branch := range branches {
		if len(branch.Tokens) == 0 {
			return 0, fmt.Errorf("branch %d has no tokens", i)
		}
		if len(branch.CandidateTokens) == 0 {
			return 0, fmt.Errorf("branch %d has no candidate tokens", i)
		}
		maxDepth = max(maxDepth, len(branch.Tokens))
		for j, token := range append(append([]int(nil), branch.Tokens...), branch.CandidateTokens...) {
			if token < 0 || token >= m.Config.VocabSize {
				return 0, fmt.Errorf("branch %d token[%d]=%d outside vocab=%d", i, j, token, m.Config.VocabSize)
			}
		}
	}
	if len(prompt)+maxDepth > MaxContextTokens {
		return 0, fmt.Errorf("prepared prompt %d + branch %d exceeds QEV context %d", len(prompt), maxDepth, MaxContextTokens)
	}
	if maxContext := m.Config.MaxSeqLen; maxContext > 0 && len(prompt)+maxDepth > maxContext {
		return 0, fmt.Errorf("prepared prompt %d + branch %d exceeds model context %d", len(prompt), maxDepth, maxContext)
	}
	if maxBranchTokens != nil {
		*maxBranchTokens = maxDepth
	}
	return maxDepth, nil
}

func prepareGemma4ScoringSession(ctx context.Context, m *model.LlamaModel, backend model.InferenceBackend, promptCache model.Gemma4PromptCacheConfig, prompt []int, branches []Branch, allowCache bool) (*model.Gemma4DecodeSession, error) {
	maxBranchTokens, err := validateGemma4ScoringInput(ctx, m, prompt, branches, nil)
	if err != nil {
		return nil, err
	}
	if backend == "" {
		backend = model.InferenceBackendSIMD
	}
	if !allowCache {
		promptCache = model.Gemma4PromptCacheConfig{}
	}
	session, err := model.NewGemma4DecodeSessionWithPromptCache(m, model.SessionOptions{Backend: backend, MaxTokens: maxBranchTokens}, promptCache)
	if err != nil {
		return nil, err
	}
	ok := false
	defer func() {
		if !ok {
			_ = session.Close()
		}
	}()
	if err := session.BeginPreparedPrefill(prompt); err != nil {
		return nil, err
	}
	for session.RemainingPrefill() > 0 {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		limit := min(UBatchTokens, session.RemainingPrefill())
		if _, err := session.PrefillNext(limit); err != nil {
			return nil, err
		}
	}
	ok = true
	return session, nil
}
