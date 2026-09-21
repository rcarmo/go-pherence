package gliner2

import (
	"fmt"
	"math"
	"sort"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

// ProposalSettings mirrors the sparse boundary proposer knobs needed for
// inference. This package intentionally implements only per-query sparse
// proposals, not training-time gold injection or shared candidate pooling.
type ProposalSettings struct {
	StartTopK               int     `json:"start_top_k"`
	EndTopK                 int     `json:"end_top_k"`
	EndsPerStart            int     `json:"ends_per_start"`
	StartsPerEnd            int     `json:"starts_per_end"`
	CandidateBudget         int     `json:"candidate_budget"`
	TrainingCandidateBudget int     `json:"training_candidate_budget,omitempty"`
	MaxGoldPerQuery         int     `json:"max_gold_per_query,omitempty"`
	EndBlockSize            int     `json:"end_block_size"`
	Bidirectional           bool    `json:"bidirectional"`
	ExportMode              string  `json:"export_mode,omitempty"`
	VectorizedPairElements  int     `json:"vectorized_pair_elements,omitempty"`
	EnableRotaryEndpoints   bool    `json:"enable_rotary_endpoints,omitempty"`
	RotaryBase              float32 `json:"rotary_base,omitempty"`
	BoundaryTopKAlpha       float64 `json:"boundary_top_k_alpha,omitempty"`
	BoundaryTopKMax         int     `json:"boundary_top_k_max,omitempty"`
	BoundaryTopKBucket      int     `json:"boundary_top_k_bucket,omitempty"`
}

func (s ProposalSettings) rotaryBase() float32 {
	if s.RotaryBase == 0 {
		return 10000
	}
	return s.RotaryBase
}

func (s ProposalSettings) boundaryTopKMaxValue() int {
	if s.BoundaryTopKMax == 0 {
		return max(s.StartTopK, s.EndTopK)
	}
	return s.BoundaryTopKMax
}

func (s ProposalSettings) boundaryTopKBucketValue() int {
	if s.BoundaryTopKBucket == 0 {
		return 1
	}
	return s.BoundaryTopKBucket
}

func (s ProposalSettings) exportModeValue() string {
	if s.ExportMode == "" {
		return "auto"
	}
	return normalizeLower(s.ExportMode)
}

func (s ProposalSettings) Validate() error {
	for key, value := range map[string]int{
		"start_top_k":      s.StartTopK,
		"end_top_k":        s.EndTopK,
		"ends_per_start":   s.EndsPerStart,
		"starts_per_end":   s.StartsPerEnd,
		"candidate_budget": s.CandidateBudget,
		"end_block_size":   s.EndBlockSize,
	} {
		if value <= 0 {
			return fmt.Errorf("proposal settings %s must be > 0, got %d", key, value)
		}
	}
	if s.BoundaryTopKAlpha < 0 {
		return fmt.Errorf("proposal settings boundary_top_k_alpha must be >= 0, got %v", s.BoundaryTopKAlpha)
	}
	if s.boundaryTopKMaxValue() < max(s.StartTopK, s.EndTopK) {
		return fmt.Errorf("proposal settings boundary_top_k_max must be >= start_top_k and end_top_k")
	}
	if s.boundaryTopKBucketValue() <= 0 {
		return fmt.Errorf("proposal settings boundary_top_k_bucket must be > 0")
	}
	if mode := s.exportModeValue(); !containsOneOf(mode, "auto", "streaming", "vectorized") {
		return fmt.Errorf("proposal settings export_mode must be 'auto', 'streaming', or 'vectorized', got %q", s.ExportMode)
	}
	if s.EnableRotaryEndpoints && s.RotaryBase < 0 {
		return fmt.Errorf("proposal settings rotary_base must be > 0")
	}
	if base := s.rotaryBase(); base <= 0 || math.IsNaN(float64(base)) || math.IsInf(float64(base), 0) {
		return fmt.Errorf("proposal settings rotary_base=%g", base)
	}
	return nil
}

// ProposalSettingsFromConfig extracts sparse proposer settings from a published
// boundary checkpoint config. Shared candidate pooling is intentionally
// unsupported here; this file only implements the per-query sparse proposer.
func ProposalSettingsFromConfig(c BoundaryHeadConfig) (ProposalSettings, error) {
	pool := normalizeLower(c.CandidatePool)
	if pool != "per_query" {
		if pool == "shared" {
			return ProposalSettings{}, fmt.Errorf("boundary_head.candidate_pool=%q is unsupported by sparse proposer inference; shared pooling must be implemented separately", c.CandidatePool)
		}
		return ProposalSettings{}, fmt.Errorf("boundary_head.candidate_pool must be 'per_query', got %q", c.CandidatePool)
	}
	settings := ProposalSettings{
		StartTopK:               c.StartTopK,
		EndTopK:                 c.EndTopK,
		EndsPerStart:            c.EndsPerStart,
		StartsPerEnd:            c.StartsPerEnd,
		CandidateBudget:         c.CandidateBudget,
		TrainingCandidateBudget: c.TrainingCandidateBudget,
		MaxGoldPerQuery:         c.MaxGoldPerQuery,
		EndBlockSize:            c.EndBlockSize,
		Bidirectional:           c.BidirectionalProposals,
		ExportMode:              c.ExportMode,
		VectorizedPairElements:  c.VectorizedPairElements,
		EnableRotaryEndpoints:   c.EnableRotaryEndpoints,
		RotaryBase:              float32(c.RotaryBase),
		BoundaryTopKAlpha:       c.BoundaryTopKAlpha,
		BoundaryTopKMax:         c.BoundaryTopKMax,
		BoundaryTopKBucket:      c.BoundaryTopKBucket,
	}
	return settings, settings.Validate()
}

// BoundaryProposals is the sparse proposer output for one padded document and
// its padded schema queries. Indices are half-open [start, end) spans.
type BoundaryProposals struct {
	Indices      [][][]int   `json:"indices"`
	Logits       [][]float32 `json:"logits"`
	CompatLogits [][]float32 `json:"compat_logits"`
	ValidMask    [][]bool    `json:"valid_mask"`
}

// SparseBoundaryProposer implements boundary/proposal.py inference only. It is
// not a full extraction model and does not implement training-time gold paths.
type SparseBoundaryProposer struct {
	BoundaryDim          int              `json:"boundary_dim"`
	QueryDim             int              `json:"query_dim"`
	Settings             ProposalSettings `json:"settings"`
	StartPairProjection  Linear           `json:"start_pair_projection"`
	EndKeyProjection     Linear           `json:"end_key_projection"`
	StartQueryProjection Linear           `json:"start_query_projection"`
}

func (p SparseBoundaryProposer) Validate() error {
	if p.BoundaryDim <= 0 || p.QueryDim <= 0 {
		return fmt.Errorf("sparse proposer dims boundary=%d query=%d", p.BoundaryDim, p.QueryDim)
	}
	if err := p.Settings.Validate(); err != nil {
		return err
	}
	for name, projection := range map[string]Linear{
		"start_pair_projection":  p.StartPairProjection,
		"end_key_projection":     p.EndKeyProjection,
		"start_query_projection": p.StartQueryProjection,
	} {
		if err := projection.Validate(); err != nil {
			return fmt.Errorf("sparse proposer %s: %w", name, err)
		}
	}
	if p.StartPairProjection.InDim != p.BoundaryDim || p.StartPairProjection.OutDim != p.BoundaryDim {
		return fmt.Errorf("sparse proposer start_pair_projection dims out=%d in=%d want out=%d in=%d", p.StartPairProjection.OutDim, p.StartPairProjection.InDim, p.BoundaryDim, p.BoundaryDim)
	}
	if p.EndKeyProjection.InDim != p.BoundaryDim || p.EndKeyProjection.OutDim != p.BoundaryDim {
		return fmt.Errorf("sparse proposer end_key_projection dims out=%d in=%d want out=%d in=%d", p.EndKeyProjection.OutDim, p.EndKeyProjection.InDim, p.BoundaryDim, p.BoundaryDim)
	}
	gateDim := p.BoundaryDim
	if p.Settings.EnableRotaryEndpoints {
		if p.BoundaryDim%2 != 0 {
			return fmt.Errorf("sparse proposer rotary endpoints require even boundary_dim=%d", p.BoundaryDim)
		}
		gateDim = p.BoundaryDim / 2
	}
	if p.StartQueryProjection.InDim != p.QueryDim || p.StartQueryProjection.OutDim != gateDim {
		return fmt.Errorf("sparse proposer start_query_projection dims out=%d in=%d want out=%d in=%d", p.StartQueryProjection.OutDim, p.StartQueryProjection.InDim, gateDim, p.QueryDim)
	}
	return nil
}

// LoadBoundaryProposer binds the sparse proposer tensors from a published
// checkpoint. Only per-query candidate pools are supported here.
func LoadBoundaryProposer(source TensorSource, queryDim int, c BoundaryHeadConfig) (SparseBoundaryProposer, error) {
	var p SparseBoundaryProposer
	if source == nil || queryDim <= 0 {
		return p, fmt.Errorf("tensor source and query width required")
	}
	settings, err := ProposalSettingsFromConfig(c)
	if err != nil {
		return p, err
	}
	r := weightReader{source: source}
	gateDim := c.BoundaryDim
	if settings.EnableRotaryEndpoints {
		gateDim /= 2
	}
	prefix := "boundary_head.boundary_proposer"
	p = SparseBoundaryProposer{
		BoundaryDim:          c.BoundaryDim,
		QueryDim:             queryDim,
		Settings:             settings,
		StartPairProjection:  r.linear(prefix+".start_pair_projection", c.BoundaryDim, c.BoundaryDim),
		EndKeyProjection:     r.linear(prefix+".end_key_projection", c.BoundaryDim, c.BoundaryDim),
		StartQueryProjection: r.linear(prefix+".start_query_projection", queryDim, gateDim),
	}
	if r.err != nil {
		return SparseBoundaryProposer{}, r.err
	}
	return p, p.Validate()
}

// ScoreExplicitPairs returns the proposer compatibility term for caller-provided
// half-open [start, end) pairs. Invalid slots are zeroed.
func (p SparseBoundaryProposer) ScoreExplicitPairs(boundaryStates, queryStates [][]float32, indices [][][]int, validMask [][]bool) ([][]float32, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}
	if len(indices) != len(queryStates) || len(validMask) != len(indices) {
		return nil, fmt.Errorf("explicit pair query shape mismatch")
	}
	startAll, endAll, gates, err := p.prepare(boundaryStates, queryStates)
	if err != nil {
		return nil, err
	}
	compat := make([][]float32, len(indices))
	scale := float32(1 / math.Sqrt(float64(p.BoundaryDim)))
	gated := make([]float32, p.BoundaryDim)
	for q := range indices {
		if len(indices[q]) != len(validMask[q]) {
			return nil, fmt.Errorf("explicit pair valid_mask[%d] len=%d want=%d", q, len(validMask[q]), len(indices[q]))
		}
		compat[q] = make([]float32, len(indices[q]))
		for c := range indices[q] {
			if len(indices[q][c]) != 2 {
				return nil, fmt.Errorf("explicit pair indices[%d][%d] len=%d want=2", q, c, len(indices[q][c]))
			}
			if !validMask[q][c] {
				continue
			}
			start := clampIndex(indices[q][c][0], len(startAll))
			end := clampIndex(indices[q][c][1], len(endAll))
			gateRow := gates[q]
			for i := 0; i < p.BoundaryDim; i++ {
				gated[i] = startAll[start][i] * gateRow[i]
			}
			compat[q][c] = simd.Sdot(gated, endAll[end]) * scale
		}
	}
	return compat, nil
}

// Forward runs the sparse proposer for one padded document and padded schema
// queries. It returns a fixed candidate budget per query.
func (p SparseBoundaryProposer) Forward(boundaryStates, queryStates [][]float32, boundaryMask, queryMask []bool, startLogits, endLogits [][]float32) (BoundaryProposals, error) {
	if err := p.Validate(); err != nil {
		return BoundaryProposals{}, err
	}
	if len(boundaryStates) == 0 {
		return BoundaryProposals{}, fmt.Errorf("sparse proposer requires at least one boundary state")
	}
	if len(boundaryStates) != len(boundaryMask) {
		return BoundaryProposals{}, fmt.Errorf("boundary mask len=%d want=%d", len(boundaryMask), len(boundaryStates))
	}
	if len(queryStates) != len(queryMask) || len(startLogits) != len(queryStates) || len(endLogits) != len(queryStates) {
		return BoundaryProposals{}, fmt.Errorf("query mask/logit shape mismatch")
	}
	for q := range queryStates {
		if len(startLogits[q]) != len(boundaryStates) {
			return BoundaryProposals{}, fmt.Errorf("start_logits[%d] len=%d want=%d", q, len(startLogits[q]), len(boundaryStates))
		}
		if len(endLogits[q]) != len(boundaryStates) {
			return BoundaryProposals{}, fmt.Errorf("end_logits[%d] len=%d want=%d", q, len(endLogits[q]), len(boundaryStates))
		}
	}
	startAll, endAll, gates, err := p.prepare(boundaryStates, queryStates)
	if err != nil {
		return BoundaryProposals{}, err
	}
	settings := p.Settings
	n := len(boundaryStates)
	startK := resolveBoundaryBudget(n, settings.StartTopK, settings.BoundaryTopKAlpha, settings.boundaryTopKMaxValue(), settings.boundaryTopKBucketValue())
	endK := resolveBoundaryBudget(n, settings.EndTopK, settings.BoundaryTopKAlpha, settings.boundaryTopKMaxValue(), settings.boundaryTopKBucketValue())
	scale := float32(1 / math.Sqrt(float64(p.BoundaryDim)))
	out := newProposalOutput(len(queryStates), settings.CandidateBudget)
	for q := range queryStates {
		validBoundaries := make([]bool, n)
		for i := 0; i < n; i++ {
			validBoundaries[i] = boundaryMask[i] && queryMask[q]
		}
		stScores, stIdx, stValid, err := selectTopBoundaries(startLogits[q], validBoundaries, startK)
		if err != nil {
			return BoundaryProposals{}, err
		}
		allStarts := make([]int, 0, len(stIdx)*settings.EndsPerStart+len(stIdx)*settings.StartsPerEnd)
		allEnds := make([]int, 0, len(stIdx)*settings.EndsPerStart+len(stIdx)*settings.StartsPerEnd)
		allScores := make([]float32, 0, cap(allStarts))
		allValid := make([]bool, 0, cap(allStarts))
		gated := make([]float32, p.BoundaryDim)
		for slot := range stIdx {
			topScores, topEnds := scoreEndsBlockwise(startAll, endAll, gates[q], boundaryMask, queryMask[q], endLogits[q], stIdx[slot], stScores[slot], stValid[slot], settings.EndBlockSize, settings.EndsPerStart, scale, gated)
			for i := range topEnds {
				allStarts = append(allStarts, stIdx[slot])
				allEnds = append(allEnds, topEnds[i])
				allScores = append(allScores, topScores[i])
				allValid = append(allValid, stValid[slot] && queryMask[q] && topEnds[i] > stIdx[slot] && topEnds[i] >= 0 && topEnds[i] < len(boundaryMask) && boundaryMask[topEnds[i]])
			}
		}
		if settings.Bidirectional {
			enScores, enIdx, enValid, err := selectTopBoundaries(endLogits[q], validBoundaries, endK)
			if err != nil {
				return BoundaryProposals{}, err
			}
			for slot := range enIdx {
				topScores, topStarts := scoreStartsBlockwise(startAll, endAll, gates[q], boundaryMask, queryMask[q], startLogits[q], enIdx[slot], enScores[slot], enValid[slot], settings.EndBlockSize, settings.StartsPerEnd, scale, gated)
				for i := range topStarts {
					allStarts = append(allStarts, topStarts[i])
					allEnds = append(allEnds, enIdx[slot])
					allScores = append(allScores, topScores[i])
					allValid = append(allValid, enValid[slot] && queryMask[q] && topStarts[i] >= 0 && topStarts[i] < len(boundaryMask) && boundaryMask[topStarts[i]] && enIdx[slot] > topStarts[i])
				}
			}
		}
		indices, valid := deduplicateCandidates(allStarts, allEnds, allScores, allValid, settings.CandidateBudget, n)
		out.Indices[q] = indices
		out.ValidMask[q] = valid
		for c := 0; c < settings.CandidateBudget; c++ {
			if !valid[c] {
				out.Logits[q][c] = MaskLogit
				continue
			}
			start, end := indices[c][0], indices[c][1]
			for i := 0; i < p.BoundaryDim; i++ {
				gated[i] = startAll[start][i] * gates[q][i]
			}
			compat := simd.Sdot(gated, endAll[end]) * scale
			out.CompatLogits[q][c] = compat
			out.Logits[q][c] = compat + startLogits[q][start] + endLogits[q][end]
		}
	}
	return out, nil
}

func (p SparseBoundaryProposer) prepare(boundaryStates, queryStates [][]float32) ([][]float32, [][]float32, [][]float32, error) {
	if len(boundaryStates) == 0 {
		return nil, nil, nil, fmt.Errorf("empty boundary states")
	}
	startAll, err := projectRows(p.StartPairProjection, boundaryStates, "sparse proposer boundary states")
	if err != nil {
		return nil, nil, nil, err
	}
	endAll, err := projectRows(p.EndKeyProjection, boundaryStates, "sparse proposer boundary states")
	if err != nil {
		return nil, nil, nil, err
	}
	if p.Settings.EnableRotaryEndpoints {
		positions := make([]int, len(boundaryStates))
		for i := range positions {
			positions[i] = i
		}
		startAll, err = RotaryBoundary(startAll, positions, p.Settings.rotaryBase())
		if err != nil {
			return nil, nil, nil, err
		}
		endAll, err = RotaryBoundary(endAll, positions, p.Settings.rotaryBase())
		if err != nil {
			return nil, nil, nil, err
		}
	}
	gateRaw, err := projectRows(p.StartQueryProjection, queryStates, "sparse proposer query states")
	if err != nil {
		return nil, nil, nil, err
	}
	gates := make([][]float32, len(gateRaw))
	for q := range gateRaw {
		if p.Settings.EnableRotaryEndpoints {
			gates[q] = make([]float32, p.BoundaryDim)
			for i, v := range gateRaw[q] {
				s := Sigmoid(v)
				gates[q][2*i] = s
				gates[q][2*i+1] = s
			}
			continue
		}
		gates[q] = make([]float32, len(gateRaw[q]))
		for i, v := range gateRaw[q] {
			gates[q][i] = Sigmoid(v)
		}
	}
	return startAll, endAll, gates, nil
}

func projectRows(p Linear, rows [][]float32, name string) ([][]float32, error) {
	flat, err := flattenRows(rows, p.InDim, name)
	if err != nil {
		return nil, err
	}
	out := make([]float32, len(rows)*p.OutDim)
	if err := p.ApplyBatch(flat, out, len(rows)); err != nil {
		return nil, err
	}
	return rowsFromFlat(out, len(rows), p.OutDim), nil
}

func resolveBoundaryBudget(nBoundaries, baseK int, alpha float64, kMax, bucket int) int {
	if alpha <= 0 {
		return baseK
	}
	requested := int(math.Ceil(alpha * float64(max(nBoundaries-1, 0))))
	if requested < baseK {
		requested = baseK
	}
	if requested > kMax {
		requested = kMax
	}
	return min(kMax, int(math.Ceil(float64(requested)/float64(bucket)))*bucket)
}

func selectTopBoundaries(logits []float32, validMask []bool, k int) ([]float32, []int, []bool, error) {
	if len(logits) != len(validMask) {
		return nil, nil, nil, fmt.Errorf("select top boundaries len mismatch logits=%d mask=%d", len(logits), len(validMask))
	}
	k = min(k, len(logits))
	order := make([]int, len(logits))
	for i := range order {
		order[i] = i
	}
	masked := func(i int) float32 {
		if !validMask[i] {
			return MaskLogit
		}
		return logits[i]
	}
	sort.SliceStable(order, func(i, j int) bool { return masked(order[i]) > masked(order[j]) })
	scores := make([]float32, k)
	indices := make([]int, k)
	valid := make([]bool, k)
	for i := 0; i < k; i++ {
		idx := order[i]
		valid[i] = validMask[idx]
		if valid[i] {
			scores[i] = logits[idx]
			indices[i] = idx
		}
	}
	return scores, indices, valid, nil
}

func mergeRunningTopK(currentScores []float32, currentIndices []int, blockScores []float32, blockIndices []int, k int) ([]float32, []int) {
	totalScores := append(append(make([]float32, 0, len(currentScores)+len(blockScores)), currentScores...), blockScores...)
	totalIndices := append(append(make([]int, 0, len(currentIndices)+len(blockIndices)), currentIndices...), blockIndices...)
	order := make([]int, len(totalScores))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(i, j int) bool { return totalScores[order[i]] > totalScores[order[j]] })
	take := min(k, len(order))
	topScores := make([]float32, take)
	topIndices := make([]int, take)
	for i := 0; i < take; i++ {
		pos := order[i]
		topScores[i] = totalScores[pos]
		topIndices[i] = totalIndices[pos]
	}
	return topScores, topIndices
}

func scoreEndsBlockwise(startAll, endAll [][]float32, gate []float32, boundaryMask []bool, queryValid bool, endMarginals []float32, startIndex int, startScore float32, startValid bool, blockSize, topK int, scale float32, gated []float32) ([]float32, []int) {
	for i := range gated {
		gated[i] = 0
	}
	if startIndex >= 0 && startIndex < len(startAll) {
		for i := range gated {
			gated[i] = startAll[startIndex][i] * gate[i]
		}
	}
	topScores := make([]float32, topK)
	topIndices := make([]int, topK)
	for i := range topScores {
		topScores[i] = MaskLogit
	}
	for j0 := 0; j0 < len(endAll); j0 += blockSize {
		j1 := min(j0+blockSize, len(endAll))
		blockScores := make([]float32, j1-j0)
		blockIndices := make([]int, j1-j0)
		for j := j0; j < j1; j++ {
			blockIndices[j-j0] = j
			keep := queryValid && startValid && boundaryMask[j] && j > startIndex
			if !keep {
				blockScores[j-j0] = MaskLogit
				continue
			}
			blockScores[j-j0] = simd.Sdot(gated, endAll[j])*scale + endMarginals[j] + startScore
		}
		topScores, topIndices = mergeRunningTopK(topScores, topIndices, blockScores, blockIndices, topK)
	}
	return topScores, topIndices
}

func scoreStartsBlockwise(startAll, endAll [][]float32, gate []float32, boundaryMask []bool, queryValid bool, startMarginals []float32, endIndex int, endScore float32, endValid bool, blockSize, topK int, scale float32, gated []float32) ([]float32, []int) {
	for i := range gated {
		gated[i] = 0
	}
	if endIndex >= 0 && endIndex < len(endAll) {
		for i := range gated {
			gated[i] = endAll[endIndex][i] * gate[i]
		}
	}
	topScores := make([]float32, topK)
	topIndices := make([]int, topK)
	for i := range topScores {
		topScores[i] = MaskLogit
	}
	for i0 := 0; i0 < len(startAll); i0 += blockSize {
		i1 := min(i0+blockSize, len(startAll))
		blockScores := make([]float32, i1-i0)
		blockIndices := make([]int, i1-i0)
		for i := i0; i < i1; i++ {
			blockIndices[i-i0] = i
			keep := queryValid && endValid && boundaryMask[i] && i < endIndex
			if !keep {
				blockScores[i-i0] = MaskLogit
				continue
			}
			blockScores[i-i0] = simd.Sdot(gated, startAll[i])*scale + startMarginals[i] + endScore
		}
		topScores, topIndices = mergeRunningTopK(topScores, topIndices, blockScores, blockIndices, topK)
	}
	return topScores, topIndices
}

type proposalCandidate struct {
	start int
	end   int
	score float32
}

func deduplicateCandidates(starts, ends []int, scores []float32, valid []bool, capacity, nBoundaries int) ([][]int, []bool) {
	indices := make([][]int, capacity)
	mask := make([]bool, capacity)
	for i := range indices {
		indices[i] = []int{0, 0}
	}
	best := make(map[int]float32)
	for i := range starts {
		if !valid[i] {
			continue
		}
		key := starts[i]*nBoundaries + ends[i]
		if score, ok := best[key]; !ok || scores[i] > score {
			best[key] = scores[i]
		}
	}
	pairs := make([]proposalCandidate, 0, len(best))
	for key, score := range best {
		start := key / nBoundaries
		pairs = append(pairs, proposalCandidate{start: start, end: key - start*nBoundaries, score: score})
	}
	sort.Slice(pairs, func(i, j int) bool {
		if pairs[i].score != pairs[j].score {
			return pairs[i].score > pairs[j].score
		}
		if pairs[i].start != pairs[j].start {
			return pairs[i].start < pairs[j].start
		}
		return pairs[i].end < pairs[j].end
	})
	for i := 0; i < min(capacity, len(pairs)); i++ {
		indices[i][0] = pairs[i].start
		indices[i][1] = pairs[i].end
		mask[i] = true
	}
	return indices, mask
}

func newProposalOutput(queries, budget int) BoundaryProposals {
	out := BoundaryProposals{
		Indices:      make([][][]int, queries),
		Logits:       make([][]float32, queries),
		CompatLogits: make([][]float32, queries),
		ValidMask:    make([][]bool, queries),
	}
	for q := 0; q < queries; q++ {
		out.Indices[q] = make([][]int, budget)
		out.Logits[q] = make([]float32, budget)
		out.CompatLogits[q] = make([]float32, budget)
		out.ValidMask[q] = make([]bool, budget)
		for c := 0; c < budget; c++ {
			out.Indices[q][c] = []int{0, 0}
			out.Logits[q][c] = MaskLogit
		}
	}
	return out
}

func clampIndex(i, n int) int {
	if n <= 0 {
		return 0
	}
	if i < 0 {
		return 0
	}
	if i >= n {
		return n - 1
	}
	return i
}
