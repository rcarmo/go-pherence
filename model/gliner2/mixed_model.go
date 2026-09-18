package gliner2

import "fmt"

// MixedScores preserves schema order and maps groups into the single shared
// extractive query axis. Classification groups have no extractive query IDs.
// Relation groups also receive contextual pair scores. Record group logits
// remain raw field scores until record metadata is supplied.
type MixedScores struct {
	Input           MixedInput
	Extraction      *EntityScores
	GroupQueryIDs   [][]int
	Classifications map[int]ClassificationScores
	Relations       map[int]RelationScores
}

func (m *EntityModel) ScoreSchemas(text string, schemas []TextSchema, maxTokens int) (MixedScores, error) {
	if m == nil || m.Tokenizer == nil {
		return MixedScores{}, fmt.Errorf("uninitialised mixed model")
	}
	for _, s := range schemas {
		if s.Marker == "[R]" && (len(s.Labels) != 2 || s.Labels[0] != "head" || s.Labels[1] != "tail" || m.Relation == nil) {
			return MixedScores{}, fmt.Errorf("relation schema requires loaded scorer and ordered head/tail roles")
		}
	}
	input, err := m.Tokenizer.PrepareSchemas(text, schemas, maxTokens)
	if err != nil {
		return MixedScores{}, err
	}
	mask := make([]bool, len(input.IDs))
	for i := range mask {
		mask[i] = true
	}
	hidden, err := m.Encoder.Encode(input.IDs, mask)
	if err != nil {
		return MixedScores{}, err
	}
	out := MixedScores{Input: input, GroupQueryIDs: make([][]int, len(schemas)), Classifications: map[int]ClassificationScores{}}
	var queries [][]float32
	var labels []string
	var positions []int
	for i, g := range input.Groups {
		rows := make([][]float32, len(g.QueryPositions))
		for j, p := range g.QueryPositions {
			rows[j] = hidden[p]
		}
		if g.Schema.Marker == "[L]" {
			logits, err := m.Classifier.Forward(rows)
			if err != nil {
				return MixedScores{}, err
			}
			temp := m.Config.BoundaryHead.ClassificationTemperature
			if temp <= 0 {
				return MixedScores{}, fmt.Errorf("invalid classification temperature")
			}
			probs := make([]float64, len(logits))
			for j, v := range logits {
				probs[j] = sigmoid(float64(v) / temp)
			}
			out.Classifications[i] = ClassificationScores{Task: g.Schema.Parent, Labels: append([]string(nil), g.Schema.Labels...), Logits: logits, Probabilities: probs}
		} else {
			for j, row := range rows {
				out.GroupQueryIDs[i] = append(out.GroupQueryIDs[i], len(queries))
				queries = append(queries, row)
				labels = append(labels, g.Schema.Labels[j])
				positions = append(positions, g.QueryPositions[j])
			}
		}
	}
	if len(queries) == 0 {
		return out, nil
	}
	if len(input.Words) == 0 {
		return MixedScores{}, fmt.Errorf("extractive schemas require text words")
	}
	words := make([][]float32, len(input.TextPositions))
	for i, p := range input.TextPositions {
		words[i] = hidden[p]
	}
	tm, qm := make([]bool, len(words)), make([]bool, len(queries))
	for i := range tm {
		tm[i] = true
	}
	for i := range qm {
		qm[i] = true
	}
	boundary, err := m.Boundary.Forward(words, len(words))
	if err != nil {
		return MixedScores{}, err
	}
	marg, err := m.Queries.Forward(boundary.States, words, queries, boundary.Mask, tm, qm)
	if err != nil {
		return MixedScores{}, err
	}
	pool, err := m.Pool.Forward(boundary.States, boundary.Mask, qm, marg.StartLogits, marg.EndLogits)
	if err != nil {
		return MixedScores{}, err
	}
	logits, _, err := m.Scorer.Forward(boundary.States, queries, qm, pool, marg, len(words), words, tm)
	if err != nil {
		return MixedScores{}, err
	}
	result := &EntityScores{Input: EntityInput{IDs: input.IDs, Words: input.Words, TextPositions: input.TextPositions, QueryPositions: positions, Labels: labels}, Candidates: pool, Logits: logits}
	for i, p := range []*Linear{m.Null, m.Count} {
		if p != nil {
			flat, err := flattenRows(queries, p.InDim, "mixed optional head")
			if err != nil {
				return MixedScores{}, err
			}
			v := make([]float32, len(queries))
			if err = p.ApplyBatch(flat, v, len(queries)); err != nil {
				return MixedScores{}, err
			}
			if i == 0 {
				result.NullLogits = v
			} else {
				result.CountLogits = v
			}
		}
	}
	out.Extraction = result
	out.Relations = make(map[int]RelationScores)
	for i, g := range input.Groups {
		if g.Schema.Marker != "[R]" {
			continue
		}
		ids := out.GroupQueryIDs[i]
		settings, err := RelationProposalSettingsFromConfig(m.Config.BoundaryHead)
		if err != nil {
			return MixedScores{}, err
		}
		generator := TypedRelationPairGenerator{Settings: settings}
		pairs, err := generator.Generate(*result, []RelationTypeSpec{{RelationType: g.Schema.Parent, HeadQueryIDs: []int{ids[0]}, TailQueryIDs: []int{ids[1]}}})
		if err != nil {
			return MixedScores{}, err
		}
		var state []float32
		if m.Config.BoundaryHead.DirectionalRelationStates {
			state = append(append([]float32(nil), queries[ids[0]]...), queries[ids[1]]...)
		} else {
			state = make([]float32, len(queries[ids[0]]))
			for d := range state {
				state[d] = (queries[ids[0]][d] + queries[ids[1]][d]) * .5
			}
		}
		scores, err := m.Relation.Forward(words, [][]float32{state}, pairs)
		if err != nil {
			return MixedScores{}, err
		}
		out.Relations[i] = RelationScores{Input: result.Input, Pairs: pairs, Logits: scores}
	}
	return out, nil
}
