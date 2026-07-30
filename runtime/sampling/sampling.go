package sampling

import (
	"container/heap"
	"math"
	"math/rand"
	"sort"
)

// candidate pairs a token ID with its raw logit value.
type candidate struct {
	idx   int32
	logit float32
}

// Sample chooses a token from logits using cfg, drawing randomness from rng.
//
// If cfg.Temperature <= 0 (greedy mode), rng is never touched and may be
// nil. Otherwise rng must be non-nil; exactly one float64 is drawn from it
// via rng.Float64() and forwarded to SampleWithDraw.
func Sample(logits []float32, cfg Config, rng *rand.Rand) (Result, error) {
	if cfg.Temperature <= 0 {
		return SampleWithDraw(logits, cfg, 0)
	}
	if rng == nil {
		return Result{}, ErrNilRand
	}
	return SampleWithDraw(logits, cfg, rng.Float64())
}

// SampleWithDraw is the deterministic sampling primitive. unitDraw is a
// caller-supplied random draw in [0, 1] (clamped if out of range); the same
// inputs always produce the same Result. See the package doc for the full
// determinism contract.
func SampleWithDraw(logits []float32, cfg Config, unitDraw float64) (Result, error) {
	if len(logits) == 0 {
		return Result{}, ErrEmptyLogits
	}
	if err := cfg.Validate(); err != nil {
		return Result{}, err
	}
	if cfg.Temperature <= 0 {
		return greedy(logits)
	}

	var (
		cands  []candidate
		sorted bool
		err    error
	)
	if cfg.TopK > 0 {
		cands, err = boundedTopK(logits, cfg.TopK)
		sorted = true
	} else {
		cands, err = fullScan(logits)
	}
	if err != nil {
		return Result{}, err
	}

	weights, total := softmaxWeights(cands, cfg.Temperature)

	if cfg.topPEnabled() {
		if !sorted {
			cands, weights = sortCandidatesWithWeights(cands, weights)
		}
		cands, weights, total = truncateTopP(cands, weights, total, cfg.TopP)
	}

	if unitDraw < 0 {
		unitDraw = 0
	} else if unitDraw > 1 {
		unitDraw = 1
	}
	tok, logit := drawFromWeights(cands, weights, total, unitDraw)
	return Result{TokenID: tok, Logit: logit, Candidates: len(cands)}, nil
}

// isExcluded reports whether v must never be selected: NaN or -Inf.
func isExcluded(v float32) bool {
	return math.IsNaN(float64(v)) || math.IsInf(float64(v), -1)
}

// greedy returns the highest-logit valid token, ties broken by ascending
// token ID (first occurrence wins since the scan only replaces on strictly
// greater logit).
func greedy(logits []float32) (Result, error) {
	bestIdx := -1
	var bestLogit float32
	valid := 0
	for i, v := range logits {
		if isExcluded(v) {
			continue
		}
		valid++
		if bestIdx == -1 || v > bestLogit {
			bestIdx = i
			bestLogit = v
		}
	}
	if bestIdx == -1 {
		return Result{}, ErrAllInvalid
	}
	return Result{TokenID: bestIdx, Logit: bestLogit, Candidates: valid}, nil
}

// fullScan collects every valid logit in ascending token-ID order, without
// sorting by logit. Used when TopK is unlimited (0) and TopP is disabled, so
// the multinomial draw can walk candidates in their natural order without
// paying for a full-vocab sort.
func fullScan(logits []float32) ([]candidate, error) {
	out := make([]candidate, 0, len(logits))
	for i, v := range logits {
		if isExcluded(v) {
			continue
		}
		out = append(out, candidate{idx: int32(i), logit: v})
	}
	if len(out) == 0 {
		return nil, ErrAllInvalid
	}
	return out, nil
}

// candidateBetter reports whether a ranks ahead of b under the canonical
// ordering: descending logit, then ascending token ID.
func candidateBetter(a, b candidate) bool {
	if a.logit != b.logit {
		return a.logit > b.logit
	}
	return a.idx < b.idx
}

// candidateHeap is a min-heap over the canonical ordering's complement: its
// root (index 0) is always the current *worst* element, so bounded top-K
// selection can evict it in O(log k) when a better candidate arrives.
type candidateHeap []candidate

func (h candidateHeap) Len() int { return len(h) }
func (h candidateHeap) Less(i, j int) bool {
	// h[i] belongs above h[j] in the min-heap (i.e. h[i] is "worse") when
	// candidateBetter(h[j], h[i]) holds.
	return candidateBetter(h[j], h[i])
}
func (h candidateHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h *candidateHeap) Push(x any)   { *h = append(*h, x.(candidate)) }
func (h *candidateHeap) Pop() any {
	old := *h
	n := len(old)
	item := old[n-1]
	*h = old[:n-1]
	return item
}

// boundedTopK selects the K highest-logit valid tokens from logits without
// sorting the full vocabulary: it maintains a size-K min-heap (keyed on the
// canonical ordering) in a single O(n log k) pass, then sorts only the K
// survivors into canonical (descending logit, ascending token ID) order.
func boundedTopK(logits []float32, k int) ([]candidate, error) {
	h := make(candidateHeap, 0, k)
	valid := 0
	for i, v := range logits {
		if isExcluded(v) {
			continue
		}
		valid++
		c := candidate{idx: int32(i), logit: v}
		if len(h) < k {
			heap.Push(&h, c)
			continue
		}
		if candidateBetter(c, h[0]) {
			h[0] = c
			heap.Fix(&h, 0)
		}
	}
	if valid == 0 {
		return nil, ErrAllInvalid
	}
	out := []candidate(h)
	sortCandidatesDesc(out)
	return out, nil
}

// sortCandidatesDesc sorts cands in place into canonical order: descending
// logit, then ascending token ID.
func sortCandidatesDesc(cands []candidate) {
	sort.Slice(cands, func(i, j int) bool { return candidateBetter(cands[i], cands[j]) })
}

// sortCandidatesWithWeights sorts cands into canonical order while keeping
// weights aligned to the same permutation.
func sortCandidatesWithWeights(cands []candidate, weights []float64) ([]candidate, []float64) {
	order := make([]int, len(cands))
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(i, j int) bool { return candidateBetter(cands[order[i]], cands[order[j]]) })
	outCands := make([]candidate, len(cands))
	outWeights := make([]float64, len(weights))
	for dst, src := range order {
		outCands[dst] = cands[src]
		outWeights[dst] = weights[src]
	}
	return outCands, outWeights
}

// softmaxWeights computes unnormalized softmax weights for cands scaled by
// temperature. It returns the weights (aligned with cands) and their sum.
//
// +Inf logits are handled deterministically: after temperature scaling
// (which preserves +Inf/-Inf for temperature > 0), any token whose scaled
// logit equals the maximum scaled logit and is +Inf receives weight 1
// (rather than the NaN that a naive exp(+Inf - +Inf) would produce); every
// other token's weight is computed normally and naturally underflows to 0
// against a +Inf maximum.
func softmaxWeights(cands []candidate, temperature float64) ([]float64, float64) {
	scaled := make([]float64, len(cands))
	maxScaled := math.Inf(-1)
	for i, c := range cands {
		s := float64(c.logit) / temperature
		scaled[i] = s
		if s > maxScaled {
			maxScaled = s
		}
	}
	weights := make([]float64, len(cands))
	var total float64
	maxIsInf := math.IsInf(maxScaled, 1)
	for i, s := range scaled {
		diff := s - maxScaled
		if maxIsInf && math.IsInf(s, 1) {
			diff = 0
		}
		w := math.Exp(diff)
		weights[i] = w
		total += w
	}
	return weights, total
}

// truncateTopP truncates the descending-ordered (cands, weights) to the
// smallest prefix whose cumulative weight reaches topP * total, always
// including the threshold-crossing token. It returns the truncated slices
// and their (re-summed) total weight for renormalized sampling.
func truncateTopP(cands []candidate, weights []float64, total, topP float64) ([]candidate, []float64, float64) {
	if len(cands) <= 1 || len(cands) != len(weights) || total <= 0 {
		return cands, weights, total
	}
	threshold := topP * total
	var cum float64
	cut := len(cands)
	for i, w := range weights {
		cum += w
		if cum >= threshold {
			cut = i + 1
			break
		}
	}
	return cands[:cut], weights[:cut], cum
}

// drawFromWeights walks cands/weights in order, returning the token whose
// cumulative weight first reaches unitDraw*total. Falls back to the last
// candidate to absorb floating-point rounding at the boundary.
func drawFromWeights(cands []candidate, weights []float64, total, unitDraw float64) (int, float32) {
	threshold := unitDraw * total
	var cum float64
	for i, w := range weights {
		// Zero-mass candidates must never win, including at unitDraw=0.
		if w <= 0 {
			continue
		}
		cum += w
		if cum >= threshold {
			return int(cands[i].idx), cands[i].logit
		}
	}
	last := cands[len(cands)-1]
	return int(last.idx), last.logit
}
