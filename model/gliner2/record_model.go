package gliner2

import (
	"fmt"
	"github.com/rcarmo/go-pherence/loader/safetensors"
	"path/filepath"
)

// RecordModel adds learned record weights to the shared extraction backbone.
type RecordModel struct {
	Base             *EntityModel
	Head             RecordHead
	CandidateEncoder Linear
}
type RecordScores struct {
	Input           EntityInput
	Group           DenseRecordGroupOutput
	CandidateLogits [][]float32
}

func LoadRecordModel(dir string) (*RecordModel, error) {
	base, err := LoadEntityModel(dir)
	if err != nil {
		return nil, err
	}
	if !base.Config.BoundaryHead.EnableRecords {
		return nil, fmt.Errorf("records disabled in checkpoint")
	}
	sf, err := safetensors.Open(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		return nil, err
	}
	defer sf.Close()
	head, err := LoadRecordHead(sf, base.Encoder.Config.HiddenSize, base.Config.BoundaryHead)
	if err != nil {
		return nil, err
	}
	r := weightReader{source: sf}
	proj := r.linear("boundary_head.candidate_encoder", 2*base.Config.BoundaryHead.BoundaryDim, base.Encoder.Config.HiddenSize)
	if r.err != nil {
		return nil, r.err
	}
	return &RecordModel{base, head, proj}, nil
}

// ScoreRecord scores one plain ordered JSON-structure schema, without
// descriptions or selection-field prefixes. Record decoding is separate.
func (m *RecordModel) ScoreRecord(text, name string, spec RecordSpec, maxTokens int) (RecordScores, error) {
	if m == nil || m.Base == nil {
		return RecordScores{}, fmt.Errorf("record model unavailable")
	}
	if err := spec.Validate(); err != nil {
		return RecordScores{}, err
	}
	if name == "" {
		return RecordScores{}, fmt.Errorf("record name required")
	}
	labels := make([]string, len(spec.Fields))
	for i, f := range spec.Fields {
		if f.QueryID != i {
			return RecordScores{}, fmt.Errorf("single-record fields require sequential query IDs")
		}
		labels[i] = f.Name
	}
	base := m.Base
	input, err := base.Tokenizer.prepareSchema(text, name, "[C]", labels, maxTokens)
	if err != nil {
		return RecordScores{}, err
	}
	if len(input.Words) == 0 {
		return RecordScores{}, fmt.Errorf("empty record text")
	}
	mask := make([]bool, len(input.IDs))
	for i := range mask {
		mask[i] = true
	}
	hidden, err := base.Encoder.Encode(input.IDs, mask)
	if err != nil {
		return RecordScores{}, err
	}
	words := make([][]float32, len(input.TextPositions))
	for i, p := range input.TextPositions {
		words[i] = hidden[p]
	}
	queries := make([][]float32, len(labels))
	for i, p := range input.QueryPositions {
		queries[i] = hidden[p]
	}
	tm, qm := make([]bool, len(words)), make([]bool, len(queries))
	for i := range tm {
		tm[i] = true
	}
	for i := range qm {
		qm[i] = true
	}
	b, err := base.Boundary.Forward(words, len(words))
	if err != nil {
		return RecordScores{}, err
	}
	marg, err := base.Queries.Forward(b.States, words, queries, b.Mask, tm, qm)
	if err != nil {
		return RecordScores{}, err
	}
	pool, err := base.Pool.Forward(b.States, b.Mask, qm, marg.StartLogits, marg.EndLogits)
	if err != nil {
		return RecordScores{}, err
	}
	logits, _, err := base.Scorer.Forward(b.States, queries, qm, pool, marg, len(words), words, tm)
	if err != nil {
		return RecordScores{}, err
	}
	endpoints := make([][]float32, len(pool.Indices))
	for i, p := range pool.Indices {
		endpoints[i] = append(append([]float32(nil), b.States[p[0]]...), b.States[p[1]]...)
	}
	states, err := projectRows(m.CandidateEncoder, endpoints, "record candidate endpoints")
	if err != nil {
		return RecordScores{}, err
	}
	for i, valid := range pool.ValidMask {
		if !valid {
			clear(states[i])
		}
	}
	group, err := m.Head.ForwardGroupDense(spec, queries, states, pool, logits, qm)
	if err != nil {
		return RecordScores{}, err
	}
	return RecordScores{input, group, logits}, nil
}
