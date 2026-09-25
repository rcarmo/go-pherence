package mojev

import "fmt"

// scoreTree shares ancestors only inside one question. A request with any tree
// larger than the reserved scratch falls back to the original branch executor.
func (s *SIMDTextScorer) scoreTree(row EncodedRow) ([][]float32, error) {
	if err := validateBranchLocalText(row, s.cpu.head); err != nil {
		return nil, err
	}
	fits := true
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
		total := len(row.State) + len(q)
		for _, c := range row.Candidates[f] {
			if err := check(c); err != nil {
				return nil, err
			}
			if len(row.State)+len(q)+len(c) > len(s.rows) {
				return nil, fmt.Errorf("mojev: branch exceeds SIMD capacity")
			}
			total += len(c)
		}
		if total > len(s.rows) {
			fits = false
		}
	}
	if !fits {
		return ScoreBranchLocalText(row, s.cpu.head, s.encodeBranch)
	}
	result := make([][]float32, len(row.Questions))
	var ends [64]int
	var state, question [512]bool
	var candidate [64][512]bool
	var masks [64][]bool
	var enabled [64]bool
	for f, q := range row.Questions {
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
		for c, ids := range row.Candidates[f] {
			appendRows(ids)
			ends[c] = count
		}
		out := s.hidden[:count*1024]
		if err := s.branch.ForwardTreeInto(out, s.rows[:count], len(row.State), len(q), ends[:len(row.Candidates[f])], s.cpu.rope, s.cpu.eps); err != nil {
			return nil, err
		}
		s.cpu.normaliseFinal(out)
		clear(state[:])
		clear(question[:])
		for i := 0; i < prefix; i++ {
			if i < len(row.State) {
				state[i] = true
			} else {
				question[i] = true
			}
		}
		start := prefix
		for c, end := range ends[:len(row.Candidates[f])] {
			clear(candidate[c][:])
			for i := start; i < end; i++ {
				candidate[c][i] = true
			}
			masks[c] = candidate[c][:count]
			enabled[c] = true
			start = end
		}
		logits, err := s.cpu.head.ScoreHidden(out, state[:count], [][]bool{question[:count]}, [][][]bool{masks[:len(row.Candidates[f])]}, [][]bool{enabled[:len(row.Candidates[f])]})
		if err != nil {
			return nil, err
		}
		result[f] = logits[0]
	}
	return result, nil
}
