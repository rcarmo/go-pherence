package gosystemone

import (
	"context"
	"fmt"

	"github.com/rcarmo/go-pherence/model"
)

// ScoreSplitContexts executes one shared decision branch for independent
// contexts. PackedTokenRows=0 retains the scalar reference for comparison.
func (s *Gemma4NVIDIAScorer) ScoreSplitContexts(ctx context.Context, shared []int, contexts [][]int, branch Branch, allowCache bool) ([][]float32, error) {
	if s == nil || s.Model == nil || s.GPU == nil || ctx == nil || len(shared) == 0 || len(contexts) == 0 || len(contexts) > MaxContexts {
		return nil, fmt.Errorf("invalid packed decision scorer/input")
	}
	budget := s.PackedTokenRows
	if budget < 0 || budget > model.Gemma4PackedRows {
		return nil, fmt.Errorf("packed token rows must be 0..%d", model.Gemma4PackedRows)
	}
	paths := make([][]int, len(contexts))
	for i, tokens := range contexts {
		if len(tokens) == 0 {
			return nil, fmt.Errorf("empty context %d", i)
		}
		prompt := append(append([]int(nil), shared...), tokens...)
		if _, err := validateGemma4ScoringInput(ctx, s.Model, prompt, []Branch{branch}, nil); err != nil {
			return nil, err
		}
		paths[i] = append(append([]int(nil), tokens...), branch.Tokens...)
	}
	out := make([][]float32, len(contexts))
	packedIndices, serialIndices := partitionContexts(contexts, len(branch.Tokens), budget, 1)
	for _, i := range serialIndices {
		tokens := contexts[i]
		scores, err := s.ScoreSplitContext(ctx, shared, tokens, []Branch{branch}, allowCache)
		if err != nil {
			return nil, err
		}
		out[i] = scores[0]
	}
	if len(packedIndices) == 0 {
		return out, nil
	}
	packedPaths := make([][]int, len(packedIndices))
	for j, i := range packedIndices {
		packedPaths[j] = paths[i]
	}
	paths = packedPaths
	s.mu.Lock()
	defer s.mu.Unlock()
	prefix := s.cached
	if !allowCache || !cacheCanReuse(shared, s.cachedTokens, len(shared), s.cachedTrunkCap, prefix != nil && prefix.CanReusePrefix()) {
		if s.cached != nil {
			s.cached.Close()
			s.cached = nil
		}
		var err error
		prefix, err = s.GPU.PrefillPreparedCapacity(ctx, shared, len(shared), 1, 1)
		if err != nil {
			return nil, err
		}
		if allowCache {
			s.cached = prefix
			s.cachedTokens = append([]int(nil), shared...)
			s.cachedTrunkCap = len(shared)
		} else {
			defer prefix.Close()
		}
	}
	for start := 0; start < len(paths); {
		end, rows := start, 0
		for end < len(paths) && len(paths[end]) <= budget-rows {
			rows += len(paths[end])
			end++
		}
		candidates := make([][]int, end-start)
		for i := range candidates {
			candidates[i] = branch.CandidateTokens
		}
		scores, err := s.GPU.ScorePrefixedPacked(ctx, prefix, paths[start:end], candidates)
		if err != nil {
			return nil, fmt.Errorf("packed contexts [%d,%d): %w", start, end, err)
		}
		for j, score := range scores {
			out[packedIndices[start+j]] = score
		}
		start = end
	}
	return out, nil
}

// ScoreSplitContextTrees packs complete context/branch groups. Every branch in
// a group sees its context but no sibling branch. Larger groups fall back.
func (s *Gemma4NVIDIAScorer) ScoreSplitContextTrees(ctx context.Context, shared []int, contexts [][]int, branches []Branch, allowCache bool) ([][][]float32, error) {
	if s == nil || s.GPU == nil || s.Model == nil || ctx == nil || len(contexts) < 1 || len(contexts) > MaxContexts || len(branches) < 1 || len(shared) < 1 {
		return nil, fmt.Errorf("invalid tree batch input")
	}
	if len(branches) == 1 {
		rows, err := s.ScoreSplitContexts(ctx, shared, contexts, branches[0], allowCache)
		if err != nil {
			return nil, err
		}
		out := make([][][]float32, len(rows))
		for i := range rows {
			out[i] = [][]float32{rows[i]}
		}
		return out, nil
	}
	budget := s.PackedTokenRows
	if budget < 0 || budget > model.Gemma4PackedRows {
		return nil, fmt.Errorf("invalid packed token rows")
	}
	cost := 0
	paths, selected := make([][]int, len(branches)), make([][]int, len(branches))
	for j, b := range branches {
		cost += len(b.Tokens)
		paths[j] = b.Tokens
		selected[j] = b.CandidateTokens
	}
	for _, c := range contexts {
		prompt := append(append([]int(nil), shared...), c...)
		if len(c) == 0 {
			return nil, fmt.Errorf("empty context")
		}
		if _, err := validateGemma4ScoringInput(ctx, s.Model, prompt, branches, nil); err != nil {
			return nil, err
		}
	}
	out := make([][][]float32, len(contexts))
	packedIndices, serialIndices := partitionContexts(contexts, cost, budget, len(branches))
	for _, i := range serialIndices {
		c := contexts[i]
		v, err := s.ScoreSplitContext(ctx, shared, c, branches, allowCache)
		if err != nil {
			return nil, err
		}
		out[i] = v
	}
	if len(packedIndices) == 0 {
		return out, nil
	}
	packedContexts := make([][]int, len(packedIndices))
	for j, i := range packedIndices {
		packedContexts[j] = contexts[i]
	}
	contexts = packedContexts
	s.mu.Lock()
	defer s.mu.Unlock()
	prefix := s.cached
	if !allowCache || !cacheCanReuse(shared, s.cachedTokens, len(shared), s.cachedTrunkCap, prefix != nil && prefix.CanReusePrefix()) {
		if s.cached != nil {
			s.cached.Close()
			s.cached = nil
		}
		var err error
		prefix, err = s.GPU.PrefillPreparedCapacity(ctx, shared, len(shared), 1, 1)
		if err != nil {
			return nil, err
		}
		if allowCache {
			s.cached = prefix
			s.cachedTokens = append([]int(nil), shared...)
			s.cachedTrunkCap = len(shared)
		} else {
			defer prefix.Close()
		}
	}
	for start := 0; start < len(contexts); {
		end, rows := start, 0
		for end < len(contexts) && len(contexts[end])+cost <= budget-rows && (end-start+1)*len(branches) <= 256 {
			rows += len(contexts[end]) + cost
			end++
		}
		scores, err := s.GPU.ScorePrefixedTreeGroups(ctx, prefix, contexts[start:end], paths, selected)
		if err != nil {
			return nil, err
		}
		for j, score := range scores {
			out[packedIndices[start+j]] = score
		}
		start = end
	}
	return out, nil
}

// partitionContexts preserves order within each execution class and returns
// original indices for output placement. A long entry cannot disable its peers.
func partitionContexts(contexts [][]int, branchRows, budget, branches int) (packed, serial []int) {
	for i, c := range contexts {
		if budget <= 0 || branches < 1 || branches > 256 || branchRows > budget || len(c) > budget-branchRows {
			serial = append(serial, i)
		} else {
			packed = append(packed, i)
		}
	}
	return packed, serial
}
