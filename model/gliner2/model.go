package gliner2

import (
	"encoding/json"
	"fmt"
	"github.com/rcarmo/go-pherence/loader/safetensors"
	"os"
	"path/filepath"
)

// EntityModel binds the published shared-pool inference components. It returns
// raw candidate scores, not the record/relation decoder's public output schema.
type EntityModel struct {
	Config      Config
	Tokenizer   *Tokenizer
	Encoder     Deberta
	Boundary    BoundaryEncoder
	Queries     BoundaryQueryHead
	Pool        DocumentCandidatePool
	Scorer      SharedPoolScorer
	Null, Count *Linear
}

type EntityScores struct {
	Input                   EntityInput
	Candidates              PooledCandidates
	Logits                  [][]float32 // candidate-major [candidate][label]
	NullLogits, CountLogits []float32
}

// LoadEntityModel loads a local published-layout directory. Tensor values are
// owned after loading; the mmap is released before returning.
func LoadEntityModel(dir string) (*EntityModel, error) {
	cf, err := os.Open(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	cfg, err := LoadConfig(cf)
	cf.Close()
	if err != nil {
		return nil, err
	}
	if cfg.BoundaryHead.CandidatePool != "shared" {
		return nil, fmt.Errorf("entity pipeline requires shared pool")
	}
	raw, err := os.ReadFile(filepath.Join(dir, "encoder_config", "config.json"))
	if err != nil {
		return nil, err
	}
	var ec DebertaConfig
	if err = json.Unmarshal(raw, &ec); err != nil {
		return nil, err
	}
	tf, err := os.Open(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		return nil, err
	}
	tok, err := LoadTokenizer(tf)
	tf.Close()
	if err != nil {
		return nil, err
	}
	weights, err := safetensors.Open(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		return nil, err
	}
	defer weights.Close()
	m := &EntityModel{Config: cfg, Tokenizer: tok}
	m.Encoder, err = LoadDeberta(weights, ec)
	if err != nil {
		return nil, err
	}
	m.Boundary, m.Queries, err = LoadBoundaryModules(weights, ec.HiddenSize, cfg.BoundaryHead)
	if err != nil {
		return nil, err
	}
	m.Pool, err = LoadDocumentCandidatePool(weights, cfg.BoundaryHead)
	if err != nil {
		return nil, err
	}
	m.Scorer, err = LoadSharedPoolScorer(weights, ec.HiddenSize, cfg.BoundaryHead)
	if err != nil {
		return nil, err
	}
	r := weightReader{source: weights}
	if cfg.BoundaryHead.EnableAbstention {
		p := r.linear("boundary_head.null_projection", ec.HiddenSize, 1)
		m.Null = &p
	}
	if cfg.BoundaryHead.EnableCountHead {
		p := r.linear("boundary_head.count_head", ec.HiddenSize, 1)
		m.Count = &p
	}
	if r.err != nil {
		return nil, r.err
	}
	return m, nil
}

func (m *EntityModel) ScoreEntities(text string, labels []string, maxTokens int) (EntityScores, error) {
	if m == nil || m.Tokenizer == nil {
		return EntityScores{}, fmt.Errorf("uninitialised entity model")
	}
	input, err := m.Tokenizer.PrepareEntities(text, labels, maxTokens)
	if err != nil {
		return EntityScores{}, err
	}
	if len(input.Words) == 0 {
		return EntityScores{}, fmt.Errorf("entity text contains no words")
	}
	mask := make([]bool, len(input.IDs))
	for i := range mask {
		mask[i] = true
	}
	hidden, err := m.Encoder.Encode(input.IDs, mask)
	if err != nil {
		return EntityScores{}, err
	}
	words := make([][]float32, len(input.TextPositions))
	for i, p := range input.TextPositions {
		words[i] = hidden[p]
	}
	queries := make([][]float32, len(input.QueryPositions))
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
	boundary, err := m.Boundary.Forward(words, len(words))
	if err != nil {
		return EntityScores{}, err
	}
	marg, err := m.Queries.Forward(boundary.States, words, queries, boundary.Mask, tm, qm)
	if err != nil {
		return EntityScores{}, err
	}
	pool, err := m.Pool.Forward(boundary.States, boundary.Mask, qm, marg.StartLogits, marg.EndLogits)
	if err != nil {
		return EntityScores{}, err
	}
	logits, _, err := m.Scorer.Forward(boundary.States, queries, qm, pool, marg, len(words), words, tm)
	if err != nil {
		return EntityScores{}, err
	}
	out := EntityScores{Input: input, Candidates: pool, Logits: logits}
	for i, p := range []*Linear{m.Null, m.Count} {
		if p != nil {
			flat, err := flattenRows(queries, p.InDim, "optional query head")
			if err != nil {
				return EntityScores{}, err
			}
			values := make([]float32, len(queries))
			if err = p.ApplyBatch(flat, values, len(queries)); err != nil {
				return EntityScores{}, err
			}
			if i == 0 {
				out.NullLogits = values
			} else {
				out.CountLogits = values
			}
		}
	}
	return out, nil
}
