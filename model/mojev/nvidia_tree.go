package mojev

import "fmt"

// fillGPUTree writes parent/local position/node-start/visible-end records. The
// CUDA tree kernels receive only validated records; -1 is the root sentinel.
func fillGPUTree(dst []uint32, n, ns, nq int, ends []int) error {
	if n < 3 || n > 512 || len(dst) != n*4 || ns < 1 || ns >= n || nq < 1 || nq >= n-ns || len(ends) < 1 || len(ends) > 64 {
		return fmt.Errorf("mojev: invalid GPU tree")
	}
	prefix := ns + nq
	start := prefix
	for _, end := range ends {
		if end <= start || end > n {
			return fmt.Errorf("mojev: invalid GPU candidate end")
		}
		start = end
	}
	if start != n {
		return fmt.Errorf("mojev: incomplete GPU tree")
	}
	for t := 0; t < prefix; t++ {
		end := prefix
		if t < ns {
			end = ns
		}
		copy(dst[t*4:], []uint32{uint32(t - 1), uint32(t), 0, uint32(end)})
	}
	start = prefix
	for _, end := range ends {
		for t := start; t < end; t++ {
			parent := t - 1
			if t == start {
				parent = prefix - 1
			}
			copy(dst[t*4:], []uint32{uint32(parent), uint32(prefix + t - start), uint32(start), uint32(end)})
		}
		start = end
	}
	return nil
}

func (g *NVIDIATextScorer) scoreTree(row EncodedRow) ([][]float32, error) {
	if err := validateBranchLocalText(row, g.cpu.head); err != nil {
		return nil, err
	}
	check := func(ids []int) error {
		for _, id := range ids {
			if id >= g.cpu.meta.VocabSize {
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
			if len(row.State)+len(q)+len(c) > g.maxTokens {
				return nil, fmt.Errorf("mojev: branch exceeds GPU token capacity")
			}
		}
	}
	result := make([][]float32, len(row.Questions))
	var ids [512]int
	var ends [64]int
	var state, question [512]bool
	var candidate [64][512]bool
	var masks [64][]bool
	var enabled [64]bool
	for f, q := range row.Questions {
		result[f] = make([]float32, len(row.Candidates[f]))
		for first := 0; first < len(row.Candidates[f]); {
			last := candidateTreeEnd(row.Candidates[f], first, len(row.State)+len(q), g.maxTokens)
			candidates := row.Candidates[f][first:last]
			n := copy(ids[:], row.State)
			n += copy(ids[n:], q)
			prefix := n
			for c, values := range candidates {
				n += copy(ids[n:], values)
				ends[c] = n
			}
			hidden, err := g.encodeTree(TextBranch{IDs: ids[:n], StateLen: len(row.State), QuestionLen: len(q)}, ends[:len(candidates)])
			if err != nil {
				return nil, err
			}
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
			for c, end := range ends[:len(candidates)] {
				clear(candidate[c][:])
				for i := start; i < end; i++ {
					candidate[c][i] = true
				}
				masks[c] = candidate[c][:n]
				enabled[c] = true
				start = end
			}
			logits, err := g.cpu.head.ScoreHidden(hidden, state[:n], [][]bool{question[:n]}, [][][]bool{masks[:len(candidates)]}, [][]bool{enabled[:len(candidates)]})
			if err != nil {
				return nil, err
			}
			copy(result[f][first:last], logits[0])
			first = last
		}
	}
	return result, nil
}
