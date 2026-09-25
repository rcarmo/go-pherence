package mojev

import (
	"fmt"
	"github.com/rcarmo/go-pherence/loader/tokenizer"
	"github.com/rcarmo/go-pherence/model/qwen"
)

// SIMDTextScorer pre-packs dense F32 projections and batches all tokens of a
// branch through SIMD matrix operations. The CPU reference remains available.
// Prepacked weights cost additional memory; scratch is bounded and serialised.
type SIMDTextScorer struct {
	cpu    *TextScorer
	branch *qwen.Qwen35SIMDBranch
}

func NewSIMDTextScorer(cpu *TextScorer, maxTokens int) (*SIMDTextScorer, error) {
	if cpu == nil {
		return nil, fmt.Errorf("mojev: nil CPU scorer")
	}
	b, e := qwen.NewQwen35SIMDBranch(cpu.model, cpu.meta, maxTokens)
	if e != nil {
		return nil, e
	}
	return &SIMDTextScorer{cpu, b}, nil
}
func (s *SIMDTextScorer) ScoreEncoded(row EncodedRow) ([][]float32, error) {
	if s == nil || s.cpu == nil || s.branch == nil {
		return nil, fmt.Errorf("mojev: nil SIMD scorer")
	}
	return ScoreBranchLocalText(row, s.cpu.head, func(b TextBranch) ([]float32, error) { return s.cpu.encodeBranchWith(b, s.branch) })
}
func (s *SIMDTextScorer) ScoreText(req TextRequest, tok *tokenizer.Tokenizer, sl, ql int) (*TextDecision, error) {
	if s == nil || s.cpu == nil {
		return nil, fmt.Errorf("mojev: nil SIMD scorer")
	}
	return s.cpu.scoreText(req, tok, sl, ql, s.ScoreEncoded)
}
