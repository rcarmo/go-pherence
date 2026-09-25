package mojev

import (
	"fmt"
	"sync"

	"github.com/rcarmo/go-pherence/loader/tokenizer"
	"github.com/rcarmo/go-pherence/model/qwen"
)

// SIMDTextScorer pre-packs dense F32 projections and batches all tokens of a
// branch through SIMD matrix operations. The CPU reference remains available.
// Prepacked weights cost additional memory; scratch is bounded and serialised.
type SIMDTextScorer struct {
	cpu    *TextScorer
	branch *qwen.Qwen35SIMDBranch
	mu     sync.Mutex
	rows   [][]float32
	hidden []float32
}

func NewSIMDTextScorer(cpu *TextScorer, maxTokens int) (*SIMDTextScorer, error) {
	if cpu == nil {
		return nil, fmt.Errorf("mojev: nil CPU scorer")
	}
	b, e := qwen.NewQwen35SIMDBranch(cpu.model, cpu.meta, maxTokens)
	if e != nil {
		return nil, e
	}
	return &SIMDTextScorer{cpu: cpu, branch: b, rows: make([][]float32, maxTokens), hidden: make([]float32, maxTokens*1024)}, nil
}
func (s *SIMDTextScorer) ScoreEncoded(row EncodedRow) ([][]float32, error) {
	if s == nil || s.cpu == nil || s.branch == nil {
		return nil, fmt.Errorf("mojev: nil SIMD scorer")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return ScoreBranchLocalText(row, s.cpu.head, s.encodeBranch)
}

// Internal borrowed output is consumed by the head while the scorer mutex is
// held. The public result remains owned, and no buffer escapes ScoreEncoded.
func (s *SIMDTextScorer) encodeBranch(b TextBranch) ([]float32, error) {
	n := len(b.IDs)
	if n > len(s.rows) {
		return nil, fmt.Errorf("mojev: branch exceeds SIMD capacity")
	}
	for i, id := range b.IDs {
		if id < 0 || id >= s.cpu.meta.VocabSize {
			return nil, fmt.Errorf("mojev: token outside vocabulary")
		}
		s.rows[i] = s.cpu.embedding[id*1024 : (id+1)*1024]
	}
	out := s.hidden[:n*1024]
	if err := s.branch.ForwardInto(out, s.rows[:n], b.StateLen, b.QuestionLen, s.cpu.rope, s.cpu.eps); err != nil {
		return nil, err
	}
	s.cpu.normaliseFinal(out)
	return out, nil
}

func (s *SIMDTextScorer) ScoreText(req TextRequest, tok *tokenizer.Tokenizer, sl, ql int) (*TextDecision, error) {
	if s == nil || s.cpu == nil {
		return nil, fmt.Errorf("mojev: nil SIMD scorer")
	}
	return s.cpu.scoreText(req, tok, sl, ql, s.ScoreEncoded)
}
