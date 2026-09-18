package gliner2

import (
	"fmt"
	"math"
	"sort"
)

// RelationProposalSettings mirrors the sparse typed relation proposer knobs.
// Zero-valued caps use the upstream defaults.
//
// This generator only proposes typed mention pairs from existing entity scores.
// It does not learn or score relations by itself.
type RelationProposalSettings struct {
	HeadsPerRelation int     `json:"heads_per_relation"`
	TailsPerRelation int     `json:"tails_per_relation"`
	PairCap          int     `json:"pair_cap"`
	Threshold        float64 `json:"threshold"`
}

func (s RelationProposalSettings) normalized() RelationProposalSettings {
	out := s
	if out.HeadsPerRelation == 0 {
		out.HeadsPerRelation = 32
	}
	if out.TailsPerRelation == 0 {
		out.TailsPerRelation = 32
	}
	if out.PairCap == 0 {
		out.PairCap = 128
	}
	return out
}

func (s RelationProposalSettings) Validate() error {
	s = s.normalized()
	for key, value := range map[string]int{
		"heads_per_relation": s.HeadsPerRelation,
		"tails_per_relation": s.TailsPerRelation,
		"pair_cap":           s.PairCap,
	} {
		if value <= 0 {
			return fmt.Errorf("relation proposal settings %s must be > 0, got %d", key, value)
		}
	}
	if math.IsNaN(s.Threshold) || math.IsInf(s.Threshold, 0) {
		return fmt.Errorf("relation proposal settings threshold=%v", s.Threshold)
	}
	if s.Threshold < 0 || s.Threshold > 1 {
		return fmt.Errorf("relation proposal settings threshold must be in [0, 1], got %v", s.Threshold)
	}
	return nil
}

// RelationProposalSettingsFromConfig extracts typed relation pair proposal
// settings from a published boundary head config.
func RelationProposalSettingsFromConfig(c BoundaryHeadConfig) (RelationProposalSettings, error) {
	s := RelationProposalSettings{
		HeadsPerRelation: c.RelationHeadsPerType,
		TailsPerRelation: c.RelationTailsPerType,
		PairCap:          c.RelationPairCap,
		Threshold:        c.RelationArgumentProposalThreshold,
	}
	return s, s.Validate()
}

// RelationTypeSpec is one explicit relation type schema for the single-document
// proposer. Head/tail query IDs refer to scores.Input.Labels positions.
type RelationTypeSpec struct {
	RelationType string `json:"relation_type"`
	HeadQueryIDs []int  `json:"head_query_ids"`
	TailQueryIDs []int  `json:"tail_query_ids"`
	AllowSelf    bool   `json:"allow_self"`
}

// RelationPair is one proposed typed relation argument pair. Priority is only
// the additive capping heuristic head_probability + tail_probability.
type RelationPair struct {
	RelationIndex      int     `json:"relation_index"`
	RelationType       string  `json:"relation_type"`
	HeadQueryID        int     `json:"head_query_id"`
	HeadCandidateIndex int     `json:"head_candidate_index"`
	HeadStart          int     `json:"head_start"`
	HeadEnd            int     `json:"head_end"`
	TailQueryID        int     `json:"tail_query_id"`
	TailCandidateIndex int     `json:"tail_candidate_index"`
	TailStart          int     `json:"tail_start"`
	TailEnd            int     `json:"tail_end"`
	HeadProbability    float64 `json:"head_probability"`
	TailProbability    float64 `json:"tail_probability"`
	Priority           float64 `json:"priority"`
}

// RelationPairProposals is the fixed-cap flattened [relation][pair] output for
// one document. PairMask marks valid proposal slots.
type RelationPairProposals struct {
	Pairs    []RelationPair `json:"pairs"`
	PairMask []bool         `json:"pair_mask"`
}

func (p RelationPairProposals) Len() int {
	return len(p.Pairs)
}

// TypedRelationPairGenerator proposes typed mention pairs from existing entity
// scores. Equal-probability ties are resolved deterministically with stable
// sorts: argument selection keeps flattened query-major then candidate order,
// and equal pair priorities keep head-major then tail-major cross-product order.
type TypedRelationPairGenerator struct {
	Settings RelationProposalSettings `json:"settings"`
}

func (g TypedRelationPairGenerator) Validate() error {
	return g.Settings.Validate()
}

func (g TypedRelationPairGenerator) Generate(scores EntityScores, relationSchema []RelationTypeSpec) (RelationPairProposals, error) {
	settings := g.Settings.normalized()
	if err := settings.Validate(); err != nil {
		return RelationPairProposals{}, err
	}
	if len(relationSchema) == 0 {
		return RelationPairProposals{}, nil
	}
	if err := validateRelationProposalInput(scores); err != nil {
		return RelationPairProposals{}, err
	}
	queryCount := len(scores.Input.Labels)
	for i, spec := range relationSchema {
		if err := validateRelationTypeSpec(spec, queryCount, i); err != nil {
			return RelationPairProposals{}, err
		}
	}
	out := RelationPairProposals{
		Pairs:    make([]RelationPair, len(relationSchema)*settings.PairCap),
		PairMask: make([]bool, len(relationSchema)*settings.PairCap),
	}
	for relationIndex, spec := range relationSchema {
		base := relationIndex * settings.PairCap
		for slot := 0; slot < settings.PairCap; slot++ {
			out.Pairs[base+slot].RelationIndex = relationIndex
			out.Pairs[base+slot].RelationType = spec.RelationType
		}
		headArgs := selectRelationArguments(scores, spec.HeadQueryIDs, settings.Threshold, settings.HeadsPerRelation)
		tailArgs := selectRelationArguments(scores, spec.TailQueryIDs, settings.Threshold, settings.TailsPerRelation)
		pairs := selectRelationPairs(spec, relationIndex, headArgs, tailArgs, settings.PairCap)
		for i := range pairs {
			out.Pairs[base+i] = pairs[i]
			out.PairMask[base+i] = true
		}
	}
	return out, nil
}

type relationArgument struct {
	QueryID        int
	CandidateIndex int
	Start          int
	End            int
	Probability    float64
}

func validateRelationProposalInput(scores EntityScores) error {
	candidateCount := len(scores.Candidates.Indices)
	if len(scores.Candidates.ValidMask) != candidateCount {
		return fmt.Errorf("relation proposal valid_mask len=%d want=%d", len(scores.Candidates.ValidMask), candidateCount)
	}
	if len(scores.Logits) != candidateCount {
		return fmt.Errorf("relation proposal logits len=%d want=%d", len(scores.Logits), candidateCount)
	}
	queryCount := len(scores.Input.Labels)
	if queryCount == 0 {
		return fmt.Errorf("relation proposal requires entity labels")
	}
	for candidateIndex := 0; candidateIndex < candidateCount; candidateIndex++ {
		if len(scores.Candidates.Indices[candidateIndex]) != 2 {
			return fmt.Errorf("relation proposal indices[%d] len=%d want=2", candidateIndex, len(scores.Candidates.Indices[candidateIndex]))
		}
		if len(scores.Logits[candidateIndex]) != queryCount {
			return fmt.Errorf("relation proposal logits[%d] len=%d want=%d", candidateIndex, len(scores.Logits[candidateIndex]), queryCount)
		}
		span := scores.Candidates.Indices[candidateIndex]
		if scores.Candidates.ValidMask[candidateIndex] && (span[0] < 0 || span[1] < span[0]) {
			return fmt.Errorf("relation proposal invalid half-open span [%d,%d) at candidate %d", span[0], span[1], candidateIndex)
		}
	}
	return nil
}

func validateRelationTypeSpec(spec RelationTypeSpec, queryCount, relationIndex int) error {
	for _, field := range []struct {
		name     string
		queryIDs []int
	}{
		{name: "head_query_ids", queryIDs: spec.HeadQueryIDs},
		{name: "tail_query_ids", queryIDs: spec.TailQueryIDs},
	} {
		for pos, queryID := range field.queryIDs {
			if queryID < 0 || queryID >= queryCount {
				return fmt.Errorf("relation schema[%d] %s[%d]=%d outside [0,%d)", relationIndex, field.name, pos, queryID, queryCount)
			}
		}
	}
	return nil
}

func selectRelationArguments(scores EntityScores, queryIDs []int, threshold float64, limit int) []relationArgument {
	if limit <= 0 || len(queryIDs) == 0 {
		return nil
	}
	allowed := make([]bool, len(scores.Input.Labels))
	for _, queryID := range queryIDs {
		allowed[queryID] = true
	}
	args := make([]relationArgument, 0, len(queryIDs)*len(scores.Candidates.Indices))
	for queryID := range scores.Input.Labels {
		if !allowed[queryID] {
			continue
		}
		for candidateIndex := range scores.Candidates.Indices {
			if !scores.Candidates.ValidMask[candidateIndex] {
				continue
			}
			probability := sigmoid(float64(scores.Logits[candidateIndex][queryID]))
			if probability < threshold {
				continue
			}
			span := scores.Candidates.Indices[candidateIndex]
			args = append(args, relationArgument{
				QueryID:        queryID,
				CandidateIndex: candidateIndex,
				Start:          span[0],
				End:            span[1],
				Probability:    probability,
			})
		}
	}
	sort.SliceStable(args, func(i, j int) bool {
		return args[i].Probability > args[j].Probability
	})
	if len(args) > limit {
		args = args[:limit]
	}
	return args
}

func selectRelationPairs(spec RelationTypeSpec, relationIndex int, headArgs, tailArgs []relationArgument, pairCap int) []RelationPair {
	if pairCap <= 0 || len(headArgs) == 0 || len(tailArgs) == 0 {
		return nil
	}
	pairs := make([]RelationPair, 0, len(headArgs)*len(tailArgs))
	for _, head := range headArgs {
		for _, tail := range tailArgs {
			if !spec.AllowSelf && head.Start == tail.Start && head.End == tail.End {
				continue
			}
			pairs = append(pairs, RelationPair{
				RelationIndex:      relationIndex,
				RelationType:       spec.RelationType,
				HeadQueryID:        head.QueryID,
				HeadCandidateIndex: head.CandidateIndex,
				HeadStart:          head.Start,
				HeadEnd:            head.End,
				TailQueryID:        tail.QueryID,
				TailCandidateIndex: tail.CandidateIndex,
				TailStart:          tail.Start,
				TailEnd:            tail.End,
				HeadProbability:    head.Probability,
				TailProbability:    tail.Probability,
				Priority:           head.Probability + tail.Probability,
			})
		}
	}
	sort.SliceStable(pairs, func(i, j int) bool {
		return pairs[i].Priority > pairs[j].Priority
	})
	if len(pairs) > pairCap {
		pairs = pairs[:pairCap]
	}
	return pairs
}
