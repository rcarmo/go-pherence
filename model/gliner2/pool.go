package gliner2

import (
	"fmt"
	"math"
	"sort"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

// PooledCandidates is the inference-only shared document pool. Indices are
// half-open [start,end) spans padded to a fixed capacity.
type PooledCandidates struct {
	Indices        [][]int   `json:"indices"`
	ProposalLogits []float32 `json:"proposal_logits"`
	CompatLogits   []float32 `json:"compat_logits"`
	ValidMask      []bool    `json:"valid_mask"`
}

// DocumentCandidatePool implements upstream boundary/pool.py document pooling
// for a single padded document and its padded schema queries. This inference
// slice builds the shared candidate pool only; pooled reranking is separate.
type DocumentCandidatePool struct {
	BoundaryDim      int    `json:"boundary_dim"`
	PoolBoundaryTopK int    `json:"pool_boundary_top_k"`
	PoolSize         int    `json:"pool_size"`
	MinPoolPerQuery  int    `json:"min_pool_per_query"`
	StartProjection  Linear `json:"start_projection"`
	EndProjection    Linear `json:"end_projection"`
}

func (p DocumentCandidatePool) Validate() error {
	if p.BoundaryDim <= 0 {
		return fmt.Errorf("document candidate pool boundary_dim=%d", p.BoundaryDim)
	}
	if p.PoolBoundaryTopK <= 0 {
		return fmt.Errorf("document candidate pool pool_boundary_top_k must be > 0, got %d", p.PoolBoundaryTopK)
	}
	if p.PoolSize <= 0 {
		return fmt.Errorf("document candidate pool pool_size must be > 0, got %d", p.PoolSize)
	}
	if p.MinPoolPerQuery < 0 {
		return fmt.Errorf("document candidate pool min_pool_per_query must be >= 0, got %d", p.MinPoolPerQuery)
	}
	if p.MinPoolPerQuery > p.PoolSize {
		return fmt.Errorf("document candidate pool min_pool_per_query=%d exceeds pool_size=%d", p.MinPoolPerQuery, p.PoolSize)
	}
	for name, projection := range map[string]Linear{
		"start_projection": p.StartProjection,
		"end_projection":   p.EndProjection,
	} {
		if err := projection.Validate(); err != nil {
			return fmt.Errorf("document candidate pool %s: %w", name, err)
		}
		if projection.InDim != p.BoundaryDim || projection.OutDim != p.BoundaryDim {
			return fmt.Errorf("document candidate pool %s dims out=%d in=%d want out=%d in=%d", name, projection.OutDim, projection.InDim, p.BoundaryDim, p.BoundaryDim)
		}
	}
	return nil
}

// LoadDocumentCandidatePool binds the shared pool builder tensors from a
// published checkpoint. The builder is materialized in upstream __init__ even
// when per-query inference is selected, so loading does not depend on
// boundary_head.candidate_pool.
func LoadDocumentCandidatePool(source TensorSource, c BoundaryHeadConfig) (DocumentCandidatePool, error) {
	var p DocumentCandidatePool
	if source == nil {
		return p, fmt.Errorf("tensor source required")
	}
	if err := c.Validate(); err != nil {
		return p, err
	}
	r := weightReader{source: source}
	prefix := "boundary_head.shared_pool_builder"
	p = DocumentCandidatePool{
		BoundaryDim:      c.BoundaryDim,
		PoolBoundaryTopK: c.PoolBoundaryTopK,
		PoolSize:         c.PoolSize,
		MinPoolPerQuery:  c.MinPoolPerQuery,
		StartProjection:  r.linear(prefix+".start_projection", c.BoundaryDim, c.BoundaryDim),
		EndProjection:    r.linear(prefix+".end_projection", c.BoundaryDim, c.BoundaryDim),
	}
	if r.err != nil {
		return DocumentCandidatePool{}, r.err
	}
	return p, p.Validate()
}

// Forward builds one shared padded pool for a single padded document.
func (p DocumentCandidatePool) Forward(boundaryStates [][]float32, boundaryMask, queryMask []bool, startLogits, endLogits [][]float32) (PooledCandidates, error) {
	if err := p.Validate(); err != nil {
		return PooledCandidates{}, err
	}
	if len(boundaryStates) == 0 {
		return PooledCandidates{}, fmt.Errorf("document candidate pool requires at least one boundary state")
	}
	n := len(boundaryStates)
	if len(boundaryMask) != n {
		return PooledCandidates{}, fmt.Errorf("boundary mask len=%d want=%d", len(boundaryMask), n)
	}
	if len(startLogits) != len(queryMask) || len(endLogits) != len(queryMask) {
		return PooledCandidates{}, fmt.Errorf("query mask/logit shape mismatch")
	}
	for q := range queryMask {
		if len(startLogits[q]) != n {
			return PooledCandidates{}, fmt.Errorf("start_logits[%d] len=%d want=%d", q, len(startLogits[q]), n)
		}
		if len(endLogits[q]) != n {
			return PooledCandidates{}, fmt.Errorf("end_logits[%d] len=%d want=%d", q, len(endLogits[q]), n)
		}
	}
	startAll, err := projectRows(p.StartProjection, boundaryStates, "document candidate pool boundary states")
	if err != nil {
		return PooledCandidates{}, err
	}
	endAll, err := projectRows(p.EndProjection, boundaryStates, "document candidate pool boundary states")
	if err != nil {
		return PooledCandidates{}, err
	}
	activeQuery := false
	for _, ok := range queryMask {
		if ok {
			activeQuery = true
			break
		}
	}
	unionStart := make([]float32, n)
	unionEnd := make([]float32, n)
	unionValid := make([]bool, n)
	for i := 0; i < n; i++ {
		unionStart[i] = MaskLogit
		unionEnd[i] = MaskLogit
		unionValid[i] = boundaryMask[i] && activeQuery
		if !boundaryMask[i] {
			continue
		}
		for q := range queryMask {
			if !queryMask[q] {
				continue
			}
			if startLogits[q][i] > unionStart[i] {
				unionStart[i] = startLogits[q][i]
			}
			if endLogits[q][i] > unionEnd[i] {
				unionEnd[i] = endLogits[q][i]
			}
		}
	}
	_, starts, startsValid, err := selectTopBoundaries(unionStart, unionValid, p.PoolBoundaryTopK)
	if err != nil {
		return PooledCandidates{}, err
	}
	_, ends, endsValid, err := selectTopBoundaries(unionEnd, unionValid, p.PoolBoundaryTopK)
	if err != nil {
		return PooledCandidates{}, err
	}
	ks, ke := len(starts), len(ends)
	pairCount := ks * ke
	pairS := make([]int, pairCount)
	pairE := make([]int, pairCount)
	pairValid := make([]bool, pairCount)
	compat := make([]float32, pairCount)
	unionPairScore := make([]float32, pairCount)
	scale := float32(1 / math.Sqrt(float64(p.BoundaryDim)))
	pos := 0
	for i := 0; i < ks; i++ {
		for j := 0; j < ke; j++ {
			start := starts[i]
			end := ends[j]
			pairS[pos] = start
			pairE[pos] = end
			valid := startsValid[i] && endsValid[j] && end > start
			pairValid[pos] = valid
			compat[pos] = simd.Sdot(startAll[clampIndex(start, len(startAll))], endAll[clampIndex(end, len(endAll))]) * scale
			unionPairScore[pos] = compat[pos] + unionStart[clampIndex(start, n)] + unionEnd[clampIndex(end, n)]
			pos++
		}
	}
	quota := min(p.MinPoolPerQuery, pairCount)
	quotaKeys := make([]int, 0, len(queryMask)*quota)
	quotaScores := make([]float32, 0, len(queryMask)*quota)
	quotaValid := make([]bool, 0, len(queryMask)*quota)
	if quota > 0 {
		rankBonus := make([]float32, quota)
		for i := 0; i < quota; i++ {
			rankBonus[i] = float32(quota - i)
		}
		priorityBase := -MaskLogit * 0.5
		for q := range queryMask {
			order := make([]int, pairCount)
			perQuery := make([]float32, pairCount)
			perQueryValid := make([]bool, pairCount)
			for i := 0; i < pairCount; i++ {
				order[i] = i
				perQueryValid[i] = pairValid[i] && queryMask[q]
				if perQueryValid[i] {
					perQuery[i] = startLogits[q][pairS[i]] + endLogits[q][pairE[i]] + compat[i]
				} else {
					perQuery[i] = MaskLogit
				}
			}
			sort.SliceStable(order, func(i, j int) bool {
				return perQuery[order[i]] > perQuery[order[j]]
			})
			for slot := 0; slot < quota; slot++ {
				idx := order[slot]
				quotaKeys = append(quotaKeys, pairS[idx]*n+pairE[idx])
				quotaScores = append(quotaScores, priorityBase+rankBonus[slot])
				quotaValid = append(quotaValid, perQueryValid[idx])
			}
		}
	}
	globalKeys := make([]int, pairCount)
	for i := 0; i < pairCount; i++ {
		globalKeys[i] = pairS[i]*n + pairE[i]
	}
	allKeys := append(quotaKeys, globalKeys...)
	allScores := append(quotaScores, unionPairScore...)
	allValid := append(quotaValid, pairValid...)
	selectedKeys, selectedValid, err := deduplicatePoolKeys(allKeys, allScores, allValid, p.PoolSize, n)
	if err != nil {
		return PooledCandidates{}, err
	}
	out := newPooledCandidates(p.PoolSize)
	for i := 0; i < p.PoolSize; i++ {
		out.ValidMask[i] = selectedValid[i]
		if !selectedValid[i] {
			continue
		}
		key := selectedKeys[i]
		start := key / n
		end := key - start*n
		out.Indices[i][0] = start
		out.Indices[i][1] = end
		comp := simd.Sdot(startAll[start], endAll[end]) * scale
		out.CompatLogits[i] = comp
		out.ProposalLogits[i] = comp + unionStart[start] + unionEnd[end]
	}
	return out, nil
}

func newPooledCandidates(capacity int) PooledCandidates {
	out := PooledCandidates{
		Indices:        make([][]int, capacity),
		ProposalLogits: make([]float32, capacity),
		CompatLogits:   make([]float32, capacity),
		ValidMask:      make([]bool, capacity),
	}
	for i := 0; i < capacity; i++ {
		out.Indices[i] = []int{0, 0}
		out.ProposalLogits[i] = MaskLogit
	}
	return out
}

type pooledKeyEntry struct {
	key   int
	score float32
	valid bool
}

func deduplicatePoolKeys(keys []int, scores []float32, valid []bool, capacity, nBoundaries int) ([]int, []bool, error) {
	if len(keys) != len(scores) || len(keys) != len(valid) {
		return nil, nil, fmt.Errorf("pool dedup length mismatch keys=%d scores=%d valid=%d", len(keys), len(scores), len(valid))
	}
	if capacity < 0 {
		return nil, nil, fmt.Errorf("pool dedup invalid capacity=%d", capacity)
	}
	if nBoundaries <= 0 {
		return nil, nil, fmt.Errorf("pool dedup invalid n_boundaries=%d", nBoundaries)
	}
	entries := make([]pooledKeyEntry, len(keys))
	invalidKey := nBoundaries * nBoundaries
	for i := range keys {
		entries[i] = pooledKeyEntry{key: invalidKey, score: MaskLogit, valid: valid[i]}
		if valid[i] {
			entries[i].key = keys[i]
			entries[i].score = scores[i]
		}
	}
	// First stable sort by descending score, then stable sort by key. This keeps
	// the highest-score occurrence first inside each key bucket, with original
	// order resolving exact-score ties before the final key-ordered tie-break.
	sort.SliceStable(entries, func(i, j int) bool { return entries[i].score > entries[j].score })
	sort.SliceStable(entries, func(i, j int) bool { return entries[i].key < entries[j].key })
	keep := make([]bool, len(entries))
	for i := range entries {
		if !entries[i].valid {
			continue
		}
		if i == 0 || entries[i].key != entries[i-1].key {
			keep[i] = true
		}
	}
	order := make([]int, len(entries))
	for i := range order {
		order[i] = i
	}
	effectiveScore := func(i int) float32 {
		if keep[i] {
			return entries[i].score
		}
		return MaskLogit
	}
	sort.SliceStable(order, func(i, j int) bool {
		return effectiveScore(order[i]) > effectiveScore(order[j])
	})
	selectedKeys := make([]int, capacity)
	selectedValid := make([]bool, capacity)
	for i := 0; i < min(capacity, len(order)); i++ {
		entry := entries[order[i]]
		selectedKeys[i] = entry.key
		selectedValid[i] = keep[order[i]]
	}
	return selectedKeys, selectedValid, nil
}
