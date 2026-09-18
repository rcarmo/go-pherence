package gliner2

import "fmt"

type RelationScores struct {
	Input  EntityInput
	Pairs  RelationPairProposals
	Logits []float32
}

// ScoreRelation handles one ordered head/tail schema. It exposes raw pair
// scores; selection and multi-relation schemas are separate decoding concerns.
func (m *EntityModel) ScoreRelation(text, relation string, maxTokens int) (RelationScores, error) {
	if m == nil || m.Tokenizer == nil || m.Relation == nil {
		return RelationScores{}, fmt.Errorf("relation model unavailable")
	}
	if relation == "" {
		return RelationScores{}, fmt.Errorf("relation name required")
	}
	input, err := m.Tokenizer.prepareSchema(text, relation, "[R]", []string{"head", "tail"}, maxTokens)
	if err != nil {
		return RelationScores{}, err
	}
	if len(input.Words) == 0 {
		return RelationScores{}, fmt.Errorf("empty relation text")
	}
	mask := make([]bool, len(input.IDs))
	for i := range mask {
		mask[i] = true
	}
	hidden, err := m.Encoder.Encode(input.IDs, mask)
	if err != nil {
		return RelationScores{}, err
	}
	words := make([][]float32, len(input.TextPositions))
	for i, p := range input.TextPositions {
		words[i] = hidden[p]
	}
	queries := [][]float32{hidden[input.QueryPositions[0]], hidden[input.QueryPositions[1]]}
	tm := make([]bool, len(words))
	for i := range tm {
		tm[i] = true
	}
	qm := []bool{true, true}
	b, err := m.Boundary.Forward(words, len(words))
	if err != nil {
		return RelationScores{}, err
	}
	marg, err := m.Queries.Forward(b.States, words, queries, b.Mask, tm, qm)
	if err != nil {
		return RelationScores{}, err
	}
	pool, err := m.Pool.Forward(b.States, b.Mask, qm, marg.StartLogits, marg.EndLogits)
	if err != nil {
		return RelationScores{}, err
	}
	logits, _, err := m.Scorer.Forward(b.States, queries, qm, pool, marg, len(words), words, tm)
	if err != nil {
		return RelationScores{}, err
	}
	settings, err := RelationProposalSettingsFromConfig(m.Config.BoundaryHead)
	if err != nil {
		return RelationScores{}, err
	}
	generator := TypedRelationPairGenerator{Settings: settings}
	pairs, err := generator.Generate(EntityScores{Input: input, Candidates: pool, Logits: logits}, []RelationTypeSpec{{RelationType: relation, HeadQueryIDs: []int{0}, TailQueryIDs: []int{1}}})
	if err != nil {
		return RelationScores{}, err
	}
	var query []float32
	if m.Config.BoundaryHead.DirectionalRelationStates {
		query = append(append([]float32(nil), queries[0]...), queries[1]...)
	} else {
		query = make([]float32, len(queries[0]))
		for i := range query {
			query[i] = (queries[0][i] + queries[1][i]) * .5
		}
	}
	// Upstream relation scorer consumes token states, despite naming its input
	// boundary_states; do not pass projected 128-wide boundary states here.
	scores, err := m.Relation.Forward(words, [][]float32{query}, pairs)
	if err != nil {
		return RelationScores{}, err
	}
	return RelationScores{input, pairs, scores}, nil
}
