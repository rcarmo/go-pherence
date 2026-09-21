package gliner2

import (
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

// SparseRelationScorer mirrors upstream boundary/relations.py sparse relation
// inference for one document's fixed-cap typed relation proposals.
type SparseRelationScorer struct {
	HiddenSize            int     `json:"hidden_size"`
	RelationQueryDim      int     `json:"relation_query_dim,omitempty"`
	UseBiaffineContent    bool    `json:"use_biaffine_content,omitempty"`
	MLPInput              Linear  `json:"mlp_input"`
	MLPOutput             Linear  `json:"mlp_output"`
	HeadContentProjection *Linear `json:"head_content_projection,omitempty"`
	TailContentProjection *Linear `json:"tail_content_projection,omitempty"`
	RelationContentGate   *Linear `json:"relation_content_gate,omitempty"`
	ContentLinear         *Linear `json:"content_linear,omitempty"`
}

func (s SparseRelationScorer) relationQueryDim() int {
	if s.RelationQueryDim > 0 {
		return s.RelationQueryDim
	}
	return s.HiddenSize
}

func (s SparseRelationScorer) Validate() error {
	if s.HiddenSize <= 0 {
		return fmt.Errorf("relation scorer hidden_size=%d", s.HiddenSize)
	}
	relationDim := s.relationQueryDim()
	if relationDim <= 0 {
		return fmt.Errorf("relation scorer relation_query_dim=%d", relationDim)
	}
	if err := s.MLPInput.Validate(); err != nil {
		return fmt.Errorf("relation scorer mlp_input: %w", err)
	}
	if err := s.MLPOutput.Validate(); err != nil {
		return fmt.Errorf("relation scorer mlp_output: %w", err)
	}
	wantIn := 4*s.HiddenSize + relationDim + 2
	if s.MLPInput.InDim != wantIn || s.MLPInput.OutDim != s.HiddenSize {
		return fmt.Errorf("relation scorer mlp_input dims out=%d in=%d want out=%d in=%d", s.MLPInput.OutDim, s.MLPInput.InDim, s.HiddenSize, wantIn)
	}
	if s.MLPOutput.InDim != s.HiddenSize || s.MLPOutput.OutDim != 1 {
		return fmt.Errorf("relation scorer mlp_output dims out=%d in=%d want out=1 in=%d", s.MLPOutput.OutDim, s.MLPOutput.InDim, s.HiddenSize)
	}
	optional := []struct {
		name string
		proj *Linear
		in   int
		out  int
	}{
		{name: "head_content_projection", proj: s.HeadContentProjection, in: s.HiddenSize, out: s.HiddenSize},
		{name: "tail_content_projection", proj: s.TailContentProjection, in: s.HiddenSize, out: s.HiddenSize},
		{name: "relation_content_gate", proj: s.RelationContentGate, in: relationDim, out: s.HiddenSize},
		{name: "content_linear", proj: s.ContentLinear, in: 2*s.HiddenSize + relationDim, out: 1},
	}
	if !s.UseBiaffineContent {
		for _, item := range optional {
			if item.proj != nil {
				return fmt.Errorf("relation scorer %s requires use_biaffine_content", item.name)
			}
		}
		return nil
	}
	for _, item := range optional {
		if item.proj == nil {
			return fmt.Errorf("relation scorer %s required", item.name)
		}
		if err := item.proj.Validate(); err != nil {
			return fmt.Errorf("relation scorer %s: %w", item.name, err)
		}
		if item.proj.InDim != item.in || item.proj.OutDim != item.out {
			return fmt.Errorf("relation scorer %s dims out=%d in=%d want out=%d in=%d", item.name, item.proj.OutDim, item.proj.InDim, item.out, item.in)
		}
	}
	return nil
}

// LoadSparseRelationScorer binds the published relation scorer weights from the
// exact relation_scorer.* safetensors prefixes.
func LoadSparseRelationScorer(source TensorSource, hiddenSize int, c BoundaryHeadConfig) (SparseRelationScorer, error) {
	var s SparseRelationScorer
	if source == nil || hiddenSize <= 0 {
		return s, fmt.Errorf("tensor source and hidden width required")
	}
	if err := c.Validate(); err != nil {
		return s, err
	}
	relationDim := hiddenSize
	if c.DirectionalRelationStates {
		relationDim = 2 * hiddenSize
	}
	r := weightReader{source: source}
	s = SparseRelationScorer{
		HiddenSize:         hiddenSize,
		RelationQueryDim:   relationDim,
		UseBiaffineContent: c.RelationBiaffineContent,
		MLPInput:           r.linear("relation_scorer.mlp.0", 4*hiddenSize+relationDim+2, hiddenSize),
		MLPOutput:          r.linear("relation_scorer.mlp.3", hiddenSize, 1),
	}
	if c.RelationBiaffineContent {
		head := r.linear("relation_scorer.head_content_projection", hiddenSize, hiddenSize)
		tail := r.linear("relation_scorer.tail_content_projection", hiddenSize, hiddenSize)
		gate := r.linear("relation_scorer.relation_content_gate", relationDim, hiddenSize)
		linear := r.linear("relation_scorer.content_linear", 2*hiddenSize+relationDim, 1)
		s.HeadContentProjection = &head
		s.TailContentProjection = &tail
		s.RelationContentGate = &gate
		s.ContentLinear = &linear
	}
	if r.err != nil {
		return SparseRelationScorer{}, r.err
	}
	return s, s.Validate()
}

// Forward scores one document's typed relation proposals. PairMask and invalid
// relation indices produce zero logits, matching upstream masked_fill behavior.
func (s SparseRelationScorer) Forward(boundaryStates, relationQueryStates [][]float32, relationPairs RelationPairProposals) ([]float32, error) {
	if err := s.Validate(); err != nil {
		return nil, err
	}
	if len(relationPairs.PairMask) != 0 && len(relationPairs.PairMask) != len(relationPairs.Pairs) {
		return nil, fmt.Errorf("relation scorer pair_mask len=%d want=%d", len(relationPairs.PairMask), len(relationPairs.Pairs))
	}
	if len(relationPairs.Pairs) == 0 {
		return []float32{}, nil
	}
	for i, row := range boundaryStates {
		if len(row) != s.HiddenSize {
			return nil, fmt.Errorf("relation scorer boundary_states[%d] len=%d want=%d", i, len(row), s.HiddenSize)
		}
	}
	relationDim := s.relationQueryDim()
	for i, row := range relationQueryStates {
		if len(row) != relationDim {
			return nil, fmt.Errorf("relation scorer relation_query_states[%d] len=%d want=%d", i, len(row), relationDim)
		}
	}
	scores := make([]float32, len(relationPairs.Pairs))
	if len(relationQueryStates) == 0 {
		return scores, nil
	}
	if len(boundaryStates) == 0 {
		return nil, fmt.Errorf("relation scorer requires boundary states")
	}
	pairValid := make([]bool, len(relationPairs.Pairs))
	features := make([]float32, len(relationPairs.Pairs)*s.MLPInput.InDim)
	relFlat := make([]float32, len(relationPairs.Pairs)*relationDim)
	length := len(boundaryStates)
	for i, pair := range relationPairs.Pairs {
		valid := pair.RelationIndex >= 0 && pair.RelationIndex < len(relationQueryStates)
		if len(relationPairs.PairMask) != 0 {
			valid = valid && relationPairs.PairMask[i]
		}
		pairValid[i] = valid
		relationIndex := clampIndex(pair.RelationIndex, len(relationQueryStates))
		rel := relationQueryStates[relationIndex]
		copy(relFlat[i*relationDim:(i+1)*relationDim], rel)
		row := features[i*s.MLPInput.InDim : (i+1)*s.MLPInput.InDim]
		pos := 0
		for _, state := range [][]float32{
			boundaryStates[clampIndex(pair.HeadStart, length)],
			boundaryStates[clampIndex(pair.HeadEnd-1, length)],
			boundaryStates[clampIndex(pair.TailStart, length)],
			boundaryStates[clampIndex(pair.TailEnd-1, length)],
			rel,
		} {
			copy(row[pos:pos+len(state)], state)
			pos += len(state)
		}
		delta := pair.TailStart - pair.HeadStart
		row[pos] = relationOrderFeature(delta)
		row[pos+1] = float32(absInt(delta)) / float32(max(length, 1))
	}
	hidden := make([]float32, len(relationPairs.Pairs)*s.HiddenSize)
	if err := s.MLPInput.ApplyBatch(features, hidden, len(relationPairs.Pairs)); err != nil {
		return nil, err
	}
	for i, v := range hidden {
		hidden[i] = gelu32(v)
	}
	if err := s.MLPOutput.ApplyBatch(hidden, scores, len(relationPairs.Pairs)); err != nil {
		return nil, err
	}
	if s.UseBiaffineContent {
		prefix := make([]float32, (len(boundaryStates)+1)*s.HiddenSize)
		for row := range boundaryStates {
			prev := prefix[row*s.HiddenSize : (row+1)*s.HiddenSize]
			next := prefix[(row+1)*s.HiddenSize : (row+2)*s.HiddenSize]
			copy(next, prev)
			for i := 0; i < s.HiddenSize; i++ {
				next[i] += boundaryStates[row][i]
			}
		}
		headMean := make([]float32, len(relationPairs.Pairs)*s.HiddenSize)
		tailMean := make([]float32, len(relationPairs.Pairs)*s.HiddenSize)
		for i, pair := range relationPairs.Pairs {
			poolRelationSpanMean(headMean[i*s.HiddenSize:(i+1)*s.HiddenSize], prefix, len(boundaryStates), s.HiddenSize, pair.HeadStart, pair.HeadEnd)
			poolRelationSpanMean(tailMean[i*s.HiddenSize:(i+1)*s.HiddenSize], prefix, len(boundaryStates), s.HiddenSize, pair.TailStart, pair.TailEnd)
		}
		headContent := make([]float32, len(headMean))
		tailContent := make([]float32, len(tailMean))
		gate := make([]float32, len(headMean))
		if err := s.HeadContentProjection.ApplyBatch(headMean, headContent, len(relationPairs.Pairs)); err != nil {
			return nil, err
		}
		if err := s.TailContentProjection.ApplyBatch(tailMean, tailContent, len(relationPairs.Pairs)); err != nil {
			return nil, err
		}
		if err := s.RelationContentGate.ApplyBatch(relFlat, gate, len(relationPairs.Pairs)); err != nil {
			return nil, err
		}
		linearIn := make([]float32, len(relationPairs.Pairs)*s.ContentLinear.InDim)
		for i := 0; i < len(relationPairs.Pairs); i++ {
			row := linearIn[i*s.ContentLinear.InDim : (i+1)*s.ContentLinear.InDim]
			copy(row, headContent[i*s.HiddenSize:(i+1)*s.HiddenSize])
			copy(row[s.HiddenSize:], tailContent[i*s.HiddenSize:(i+1)*s.HiddenSize])
			copy(row[2*s.HiddenSize:], relFlat[i*relationDim:(i+1)*relationDim])
		}
		contentScore := make([]float32, len(relationPairs.Pairs))
		if err := s.ContentLinear.ApplyBatch(linearIn, contentScore, len(relationPairs.Pairs)); err != nil {
			return nil, err
		}
		scale := float32(1 / math.Sqrt(float64(s.HiddenSize)))
		gatedHead := make([]float32, s.HiddenSize)
		for i := 0; i < len(relationPairs.Pairs); i++ {
			head := headContent[i*s.HiddenSize : (i+1)*s.HiddenSize]
			tail := tailContent[i*s.HiddenSize : (i+1)*s.HiddenSize]
			for d := 0; d < s.HiddenSize; d++ {
				gatedHead[d] = head[d] * Sigmoid(gate[i*s.HiddenSize+d])
			}
			scores[i] += simd.Sdot(gatedHead, tail)*scale + contentScore[i]
		}
	}
	for i, valid := range pairValid {
		if !valid {
			scores[i] = 0
		}
	}
	return scores, nil
}

func poolRelationSpanMean(out, prefix []float32, rows, hidden, start, end int) {
	startClamped := clampIndex(start, rows+1)
	endClamped := clampIndex(end, rows+1)
	width := float32(max(end-start, 1))
	for i := 0; i < hidden; i++ {
		out[i] = (prefix[endClamped*hidden+i] - prefix[startClamped*hidden+i]) / width
	}
}

func relationOrderFeature(delta int) float32 {
	switch {
	case delta > 0:
		return 1
	case delta < 0:
		return -1
	default:
		return 0
	}
}
