package gliner2

import (
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

// SharedPoolScorer mirrors upstream boundary/pool.py pooled reranking for one
// padded document. It returns candidate-major logits [candidate][query] and the
// shared candidate states [candidate][pair_dim]. Published GLiNER2.5 models use
// query_attention_layers=0; higher values are rejected explicitly for now.
type SharedPoolScorer struct {
	BoundaryDim          int                               `json:"boundary_dim"`
	QueryDim             int                               `json:"query_dim"`
	PairDim              int                               `json:"pair_dim"`
	QueryAttentionLayers int                               `json:"query_attention_layers,omitempty"`
	StartProjection      Linear                            `json:"start_projection"`
	EndProjection        Linear                            `json:"end_projection"`
	LengthProjection     Linear                            `json:"length_projection"`
	PriorProjection      Linear                            `json:"prior_projection"`
	ContentPooler        *SpanContentPooler                `json:"content_pooler,omitempty"`
	ContentProjection    *Linear                           `json:"content_projection,omitempty"`
	CandidateNorm        LayerNorm                         `json:"candidate_norm"`
	CandidateLayers      []OverlapBiasedCandidateAttention `json:"candidate_layers,omitempty"`
	QueryProjection      Linear                            `json:"query_projection"`
	Film                 Linear                            `json:"film"`
	FilmOutputInput      Linear                            `json:"film_output_input"`
	FilmOutputOutput     Linear                            `json:"film_output_output"`
}

func (s SharedPoolScorer) Validate() error {
	if s.BoundaryDim <= 0 || s.QueryDim <= 0 || s.PairDim <= 0 {
		return fmt.Errorf("shared pool scorer dims boundary=%d query=%d pair=%d", s.BoundaryDim, s.QueryDim, s.PairDim)
	}
	if s.QueryAttentionLayers < 0 {
		return fmt.Errorf("shared pool scorer query_attention_layers=%d", s.QueryAttentionLayers)
	}
	if s.QueryAttentionLayers > 0 {
		return fmt.Errorf("shared pool scorer query_attention_layers=%d is unsupported", s.QueryAttentionLayers)
	}
	for name, projection := range map[string]Linear{
		"start_projection":   s.StartProjection,
		"end_projection":     s.EndProjection,
		"length_projection":  s.LengthProjection,
		"prior_projection":   s.PriorProjection,
		"query_projection":   s.QueryProjection,
		"film":               s.Film,
		"film_output_input":  s.FilmOutputInput,
		"film_output_output": s.FilmOutputOutput,
	} {
		if err := projection.Validate(); err != nil {
			return fmt.Errorf("shared pool scorer %s: %w", name, err)
		}
	}
	if s.StartProjection.InDim != s.BoundaryDim || s.StartProjection.OutDim != s.PairDim {
		return fmt.Errorf("shared pool scorer start_projection dims out=%d in=%d want out=%d in=%d", s.StartProjection.OutDim, s.StartProjection.InDim, s.PairDim, s.BoundaryDim)
	}
	if s.EndProjection.InDim != s.BoundaryDim || s.EndProjection.OutDim != s.PairDim {
		return fmt.Errorf("shared pool scorer end_projection dims out=%d in=%d want out=%d in=%d", s.EndProjection.OutDim, s.EndProjection.InDim, s.PairDim, s.BoundaryDim)
	}
	if s.LengthProjection.InDim != 3 || s.LengthProjection.OutDim != s.PairDim {
		return fmt.Errorf("shared pool scorer length_projection dims out=%d in=%d want out=%d in=%d", s.LengthProjection.OutDim, s.LengthProjection.InDim, s.PairDim, 3)
	}
	if s.PriorProjection.InDim != 1 || s.PriorProjection.OutDim != s.PairDim {
		return fmt.Errorf("shared pool scorer prior_projection dims out=%d in=%d want out=%d in=%d", s.PriorProjection.OutDim, s.PriorProjection.InDim, s.PairDim, 1)
	}
	if s.ContentPooler == nil {
		if s.ContentProjection != nil {
			return fmt.Errorf("shared pool scorer content_projection requires content_pooler")
		}
	} else {
		if err := s.ContentPooler.Validate(); err != nil {
			return fmt.Errorf("shared pool scorer content_pooler: %w", err)
		}
		if s.ContentProjection == nil {
			return fmt.Errorf("shared pool scorer content_projection required when content_pooler is enabled")
		}
		if err := s.ContentProjection.Validate(); err != nil {
			return fmt.Errorf("shared pool scorer content_projection: %w", err)
		}
		if s.ContentProjection.InDim != s.ContentPooler.OutputDim() || s.ContentProjection.OutDim != s.PairDim {
			return fmt.Errorf("shared pool scorer content_projection dims out=%d in=%d want out=%d in=%d", s.ContentProjection.OutDim, s.ContentProjection.InDim, s.PairDim, s.ContentPooler.OutputDim())
		}
	}
	if err := s.CandidateNorm.Validate(); err != nil {
		return fmt.Errorf("shared pool scorer candidate_norm: %w", err)
	}
	if s.CandidateNorm.Dim() != s.PairDim {
		return fmt.Errorf("shared pool scorer candidate_norm dim=%d want=%d", s.CandidateNorm.Dim(), s.PairDim)
	}
	for i, layer := range s.CandidateLayers {
		if err := layer.Validate(); err != nil {
			return fmt.Errorf("shared pool scorer candidate_layers[%d]: %w", i, err)
		}
		if layer.Norm1.Dim() != s.PairDim {
			return fmt.Errorf("shared pool scorer candidate_layers[%d] dim=%d want=%d", i, layer.Norm1.Dim(), s.PairDim)
		}
	}
	if s.QueryProjection.InDim != s.QueryDim || s.QueryProjection.OutDim != s.PairDim {
		return fmt.Errorf("shared pool scorer query_projection dims out=%d in=%d want out=%d in=%d", s.QueryProjection.OutDim, s.QueryProjection.InDim, s.PairDim, s.QueryDim)
	}
	if s.Film.InDim != s.PairDim || s.Film.OutDim != 2*s.PairDim {
		return fmt.Errorf("shared pool scorer film dims out=%d in=%d want out=%d in=%d", s.Film.OutDim, s.Film.InDim, 2*s.PairDim, s.PairDim)
	}
	if s.FilmOutputInput.InDim != s.PairDim || s.FilmOutputInput.OutDim != 64 {
		return fmt.Errorf("shared pool scorer film_output_input dims out=%d in=%d want out=%d in=%d", s.FilmOutputInput.OutDim, s.FilmOutputInput.InDim, 64, s.PairDim)
	}
	if s.FilmOutputOutput.InDim != 64 || s.FilmOutputOutput.OutDim != 1 {
		return fmt.Errorf("shared pool scorer film_output_output dims out=%d in=%d want out=%d in=%d", s.FilmOutputOutput.OutDim, s.FilmOutputOutput.InDim, 1, 64)
	}
	return nil
}

// LoadSharedPoolScorer binds the published shared candidate reranker from the
// exact boundary_head.shared_pool_scorer.* prefixes.
func LoadSharedPoolScorer(source TensorSource, hiddenSize int, c BoundaryHeadConfig) (SharedPoolScorer, error) {
	var s SharedPoolScorer
	if source == nil || hiddenSize <= 0 {
		return s, fmt.Errorf("tensor source and hidden width required")
	}
	if err := c.Validate(); err != nil {
		return s, err
	}
	if c.QueryAttentionLayers > 0 {
		return s, fmt.Errorf("boundary_head.query_attention_layers=%d is unsupported", c.QueryAttentionLayers)
	}
	r := weightReader{source: source}
	prefix := "boundary_head.shared_pool_scorer"
	s = SharedPoolScorer{
		BoundaryDim:          c.BoundaryDim,
		QueryDim:             hiddenSize,
		PairDim:              c.PairDim,
		QueryAttentionLayers: c.QueryAttentionLayers,
		StartProjection:      r.linear(prefix+".start_projection", c.BoundaryDim, c.PairDim),
		EndProjection:        r.linear(prefix+".end_projection", c.BoundaryDim, c.PairDim),
		LengthProjection:     r.linear(prefix+".length_projection", 3, c.PairDim),
		PriorProjection:      r.linear(prefix+".prior_projection", 1, c.PairDim),
		CandidateNorm:        r.norm(prefix+".candidate_norm", c.PairDim),
		QueryProjection:      r.linear(prefix+".query_projection", hiddenSize, c.PairDim),
		Film:                 r.linear(prefix+".film", c.PairDim, 2*c.PairDim),
		FilmOutputInput:      r.linear(prefix+".film_output.0", c.PairDim, 64),
		FilmOutputOutput:     r.linear(prefix+".film_output.3", 64, 1),
	}
	if c.EnableSpanContent {
		pooler, err := LoadSpanContentPooler(source, hiddenSize, c.ContentDim, c.ContentSoftMaxPool, prefix+".content_pooler")
		if err != nil {
			return SharedPoolScorer{}, err
		}
		projection := r.linear(prefix+".content_projection", pooler.OutputDim(), c.PairDim)
		s.ContentPooler = &pooler
		s.ContentProjection = &projection
	}
	for i := 0; i < c.CandidateAttentionLayers; i++ {
		layer, err := LoadOverlapBiasedCandidateAttention(source, c.PairDim, c.CandidateAttentionHeads, fmt.Sprintf("%s.candidate_layers.%d", prefix, i))
		if err != nil {
			return SharedPoolScorer{}, err
		}
		s.CandidateLayers = append(s.CandidateLayers, layer)
	}
	if r.err != nil {
		return SharedPoolScorer{}, r.err
	}
	return s, s.Validate()
}

// Forward scores one shared padded candidate pool against one padded set of
// schema queries. Returned logits are candidate-major [candidate][query].
func (s SharedPoolScorer) Forward(boundaryStates, queryStates [][]float32, queryMask []bool, pooled PooledCandidates, marginals BoundaryMarginals, textLength int, textStates [][]float32, textMask []bool) ([][]float32, [][]float32, error) {
	if err := s.Validate(); err != nil {
		return nil, nil, err
	}
	if len(boundaryStates) == 0 {
		return nil, nil, fmt.Errorf("shared pool scorer requires at least one boundary state")
	}
	if len(queryStates) != len(queryMask) {
		return nil, nil, fmt.Errorf("shared pool scorer query_mask len=%d want=%d", len(queryMask), len(queryStates))
	}
	if textLength < 0 || textLength > len(boundaryStates)-1 {
		return nil, nil, fmt.Errorf("shared pool scorer text_length=%d outside [0,%d]", textLength, len(boundaryStates)-1)
	}
	if len(marginals.StartLogits) != len(queryStates) {
		return nil, nil, fmt.Errorf("shared pool scorer start_logits queries=%d want=%d", len(marginals.StartLogits), len(queryStates))
	}
	if len(marginals.EndLogits) != len(queryStates) {
		return nil, nil, fmt.Errorf("shared pool scorer end_logits queries=%d want=%d", len(marginals.EndLogits), len(queryStates))
	}
	for q := range queryStates {
		if len(marginals.StartLogits[q]) != len(boundaryStates) {
			return nil, nil, fmt.Errorf("shared pool scorer start_logits[%d] len=%d want=%d", q, len(marginals.StartLogits[q]), len(boundaryStates))
		}
		if len(marginals.EndLogits[q]) != len(boundaryStates) {
			return nil, nil, fmt.Errorf("shared pool scorer end_logits[%d] len=%d want=%d", q, len(marginals.EndLogits[q]), len(boundaryStates))
		}
	}
	if len(marginals.InsidePrefix) != 0 {
		if len(marginals.InsidePrefix) != len(queryStates) {
			return nil, nil, fmt.Errorf("shared pool scorer inside_prefix queries=%d want=%d", len(marginals.InsidePrefix), len(queryStates))
		}
		for q := range queryStates {
			if len(marginals.InsidePrefix[q]) == 0 {
				return nil, nil, fmt.Errorf("shared pool scorer inside_prefix[%d] empty", q)
			}
			if len(textStates) != 0 && len(marginals.InsidePrefix[q]) != len(textStates)+1 {
				return nil, nil, fmt.Errorf("shared pool scorer inside_prefix[%d] len=%d want=%d", q, len(marginals.InsidePrefix[q]), len(textStates)+1)
			}
		}
	}
	if len(marginals.InsidePrefixMean) != 0 && len(marginals.InsidePrefixMean) != len(queryStates) {
		return nil, nil, fmt.Errorf("shared pool scorer inside_prefix_mean len=%d want=%d", len(marginals.InsidePrefixMean), len(queryStates))
	}
	if len(textStates) != 0 || len(textMask) != 0 {
		if len(textStates) != len(textMask) {
			return nil, nil, fmt.Errorf("shared pool scorer text_mask len=%d want=%d", len(textMask), len(textStates))
		}
		if textLength > len(textStates) {
			return nil, nil, fmt.Errorf("shared pool scorer text_length=%d exceeds text rows=%d", textLength, len(textStates))
		}
	}
	candidateCount := len(pooled.Indices)
	if len(pooled.ValidMask) != candidateCount {
		return nil, nil, fmt.Errorf("shared pool scorer valid_mask len=%d want=%d", len(pooled.ValidMask), candidateCount)
	}
	if len(pooled.CompatLogits) != 0 && len(pooled.CompatLogits) != candidateCount {
		return nil, nil, fmt.Errorf("shared pool scorer compat_logits len=%d want=%d", len(pooled.CompatLogits), candidateCount)
	}
	if len(pooled.ProposalLogits) != 0 && len(pooled.ProposalLogits) != candidateCount {
		return nil, nil, fmt.Errorf("shared pool scorer proposal_logits len=%d want=%d", len(pooled.ProposalLogits), candidateCount)
	}
	indices := make([][2]int, candidateCount)
	starts := make([]int, candidateCount)
	ends := make([]int, candidateCount)
	for i := 0; i < candidateCount; i++ {
		if len(pooled.Indices[i]) != 2 {
			return nil, nil, fmt.Errorf("shared pool scorer indices[%d] len=%d want=2", i, len(pooled.Indices[i]))
		}
		starts[i] = pooled.Indices[i][0]
		ends[i] = pooled.Indices[i][1]
		indices[i] = [2]int{starts[i], ends[i]}
	}

	startAll, err := projectRows(s.StartProjection, boundaryStates, "shared pool scorer boundary states")
	if err != nil {
		return nil, nil, err
	}
	endAll, err := projectRows(s.EndProjection, boundaryStates, "shared pool scorer boundary states")
	if err != nil {
		return nil, nil, err
	}
	lengthFeatures := make([][]float32, candidateCount)
	priorRows := make([][]float32, candidateCount)
	prior := make([]float32, candidateCount)
	if len(pooled.CompatLogits) == candidateCount {
		copy(prior, pooled.CompatLogits)
	} else if len(pooled.ProposalLogits) == candidateCount {
		copy(prior, pooled.ProposalLogits)
	}
	for i := 0; i < candidateCount; i++ {
		features, err := ContinuousLengthFeatures(starts[i], ends[i], textLength)
		if err != nil {
			return nil, nil, err
		}
		lengthFeatures[i] = []float32{features[0], features[1], features[2]}
		priorRows[i] = []float32{prior[i]}
	}
	lengthRep, err := projectRows(s.LengthProjection, lengthFeatures, "shared pool scorer length features")
	if err != nil {
		return nil, nil, err
	}
	priorRep, err := projectRows(s.PriorProjection, priorRows, "shared pool scorer prior logits")
	if err != nil {
		return nil, nil, err
	}
	candidate := makeMatrix(candidateCount, s.PairDim)
	for i := 0; i < candidateCount; i++ {
		sIdx := clampIndex(starts[i], len(startAll))
		eIdx := clampIndex(ends[i], len(endAll))
		for d := 0; d < s.PairDim; d++ {
			candidate[i][d] = startAll[sIdx][d] + endAll[eIdx][d] + lengthRep[i][d] + priorRep[i][d]
		}
	}
	if s.ContentPooler != nil {
		meanPrefix, lsePrefix, err := s.ContentPooler.BuildPrefix(textStates, textMask)
		if err != nil {
			return nil, nil, err
		}
		content, err := s.ContentPooler.PoolRows(meanPrefix, lsePrefix, starts, ends)
		if err != nil {
			return nil, nil, err
		}
		contentRep, err := projectRows(*s.ContentProjection, content, "shared pool scorer span content")
		if err != nil {
			return nil, nil, err
		}
		for i := 0; i < candidateCount; i++ {
			for d := 0; d < s.PairDim; d++ {
				candidate[i][d] += contentRep[i][d]
			}
		}
	}
	candidateFlat, err := flattenRows(candidate, s.PairDim, "shared pool scorer candidate states")
	if err != nil {
		return nil, nil, err
	}
	normed := make([]float32, len(candidateFlat))
	if err := s.CandidateNorm.ApplyBatch(candidateFlat, normed, candidateCount); err != nil {
		return nil, nil, err
	}
	candidate = rowsFromFlat(normed, candidateCount, s.PairDim)
	for i := 0; i < candidateCount; i++ {
		if pooled.ValidMask[i] {
			continue
		}
		clear(candidate[i])
	}
	for _, layer := range s.CandidateLayers {
		candidate, err = layer.Forward(candidate, indices, pooled.ValidMask)
		if err != nil {
			return nil, nil, err
		}
	}

	query, err := projectRows(s.QueryProjection, queryStates, "shared pool scorer query states")
	if err != nil {
		return nil, nil, err
	}
	filmParams, err := projectRows(s.Film, query, "shared pool scorer projected query states")
	if err != nil {
		return nil, nil, err
	}
	logits := makeMatrix(candidateCount, len(queryStates))
	dotScale := float32(1 / math.Sqrt(float64(s.PairDim)))
	conditioned := make([]float32, s.PairDim)
	filmHidden := make([]float32, s.FilmOutputInput.OutDim)
	filmScalar := make([]float32, 1)
	for q := range query {
		gamma := filmParams[q][:s.PairDim]
		beta := filmParams[q][s.PairDim:]
		var meanValue float32
		meanProvided := len(marginals.InsidePrefixMean) == len(query)
		if meanProvided {
			meanValue = marginals.InsidePrefixMean[q]
		}
		for c := 0; c < candidateCount; c++ {
			base := simd.Sdot(candidate[c], query[q]) * dotScale
			for i := 0; i < s.PairDim; i++ {
				conditioned[i] = candidate[c][i]*(1+gamma[i]) + beta[i]
			}
			if err := s.FilmOutputInput.Apply(conditioned, filmHidden); err != nil {
				return nil, nil, err
			}
			for i := range filmHidden {
				filmHidden[i] = gelu32(filmHidden[i])
			}
			if err := s.FilmOutputOutput.Apply(filmHidden, filmScalar); err != nil {
				return nil, nil, err
			}
			start := clampIndex(starts[c], len(marginals.StartLogits[q]))
			end := clampIndex(ends[c], len(marginals.EndLogits[q]))
			logit := base + filmScalar[0] + marginals.StartLogits[q][start] + marginals.EndLogits[q][end]
			if len(marginals.InsidePrefix) != 0 {
				var meanPtr *float32
				if meanProvided {
					meanPtr = &meanValue
				}
				interval, err := IntervalPrefixScore(marginals.InsidePrefix[q], starts[c], ends[c], meanPtr)
				if err != nil {
					return nil, nil, err
				}
				logit += interval / float32(math.Sqrt(float64(max(ends[c]-starts[c], 1))))
			}
			logits[c][q] = logit
		}
	}
	for c := 0; c < candidateCount; c++ {
		for q := range queryMask {
			if !pooled.ValidMask[c] || !queryMask[q] {
				logits[c][q] = MaskLogit
			}
		}
	}
	return logits, candidate, nil
}
