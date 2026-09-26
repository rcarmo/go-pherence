package mojev

import (
	"context"
	"fmt"
)

// scoreTree shares ancestors within bounded candidate groups of one question.
// Oversized questions use multiple groups without increasing scratch capacity.
func (s *SIMDTextScorer) scoreTree(ctx context.Context, row EncodedRow) ([][]float32, error) {
	if err := validateBranchLocalText(row, s.cpu.head); err != nil {
		return nil, err
	}
	check := func(ids []int) error {
		for _, id := range ids {
			if id >= s.cpu.meta.VocabSize {
				return fmt.Errorf("mojev: token outside vocabulary")
			}
		}
		return nil
	}
	if err := check(row.State); err != nil {
		return nil, err
	}
	for f, q := range row.Questions {
		if err := check(q); err != nil {
			return nil, err
		}
		for _, c := range row.Candidates[f] {
			if err := check(c); err != nil {
				return nil, err
			}
			if len(row.State)+len(q)+len(c) > len(s.rows) {
				return nil, fmt.Errorf("mojev: branch exceeds SIMD capacity")
			}
		}
	}
	result := make([][]float32, len(row.Questions))
	var ends [64]int
	state, question := s.stateMask, s.questionMask
	stride := len(s.rows)
	if len(state) < stride || len(question) < stride || len(s.candidateMasks) < 64*stride {
		return nil, fmt.Errorf("mojev: invalid SIMD mask capacity")
	}
	var masks [64][]bool
	var enabled [64]bool
	for f, q := range row.Questions {
		result[f] = make([]float32, len(row.Candidates[f]))
		for first := 0; first < len(row.Candidates[f]); {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			last := candidateTreeEnd(row.Candidates[f], first, len(row.State)+len(q), len(s.rows))
			candidates := row.Candidates[f][first:last]
			count := 0
			appendRows := func(ids []int) {
				for _, id := range ids {
					s.rows[count] = s.cpu.embedding[id*1024 : (id+1)*1024]
					count++
				}
			}
			appendRows(row.State)
			appendRows(q)
			prefix := count
			for c, ids := range candidates {
				appendRows(ids)
				ends[c] = count
			}
			out := s.hidden[:count*1024]
			if err := s.branch.ForwardTreeIntoContext(ctx, out, s.rows[:count], len(row.State), len(q), ends[:len(candidates)], s.cpu.rope, s.cpu.eps); err != nil {
				return nil, err
			}
			s.cpu.normaliseFinal(out)
			clear(state[:count])
			clear(question[:count])
			for i := 0; i < prefix; i++ {
				if i < len(row.State) {
					state[i] = true
				} else {
					question[i] = true
				}
			}
			start := prefix
			for c, end := range ends[:len(candidates)] {
				mask := s.candidateMasks[c*stride : c*stride+count]
				clear(mask)
				for i := start; i < end; i++ {
					mask[i] = true
				}
				masks[c] = mask
				enabled[c] = true
				start = end
			}
			logits, err := s.cpu.head.ScoreHidden(out, state[:count], [][]bool{question[:count]}, [][][]bool{masks[:len(candidates)]}, [][]bool{enabled[:len(candidates)]})
			if err != nil {
				return nil, err
			}
			copy(result[f][first:last], logits[0])
			first = last
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return result, nil
}
