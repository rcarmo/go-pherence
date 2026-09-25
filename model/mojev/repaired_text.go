package mojev

import "fmt"

// TextBranch describes one isolated path in a packed text row. The encoder
// receives owned token IDs and their original packed positions; StateLen and
// QuestionLen locate the three contiguous nodes. It must create a fresh
// forward state for each call, apply the tree visibility rule to full attention,
// and keep linear-attention history local to this path. This interface does not
// implement or attest to those encoder semantics.
type TextBranch struct {
	IDs               []int
	AbsolutePositions []int
	StateLen          int
	QuestionLen       int
}

// TextBranchEncoder returns pre-head-normalisation hidden rows, flattened in
// token-major order. It must implement the separately validated repaired policy
// (bidirectional within each node for full attention, causal linear attention).
type TextBranchEncoder func(TextBranch) ([]float32, error)

// ScoreBranchLocalText runs one text-only state/question/candidate path per
// candidate and applies the MoJev head to that path. Siblings are never
// supplied to the same encoder call. This is an injectable repair boundary,
// not a native encoder or parity with the released packed checkpoint. Errors
// return no partial logits. The encoder must not carry state between calls.
func ScoreBranchLocalText(row EncodedRow, head *HeadWeights, encode TextBranchEncoder) ([][]float32, error) {
	if encode == nil || head == nil || len(row.State) == 0 || len(row.Questions) == 0 ||
		len(row.Questions) > 256 || len(row.Candidates) != len(row.Questions) {
		return nil, fmt.Errorf("mojev: invalid branch-local text request")
	}
	total := 0
	add := func(ids []int) error {
		if len(ids) == 0 || len(ids) > 4096-total {
			return fmt.Errorf("mojev: invalid branch-local text length")
		}
		for _, id := range ids {
			if id < 0 {
				return fmt.Errorf("mojev: negative token ID")
			}
		}
		total += len(ids)
		return nil
	}
	if err := add(row.State); err != nil {
		return nil, err
	}
	for f, question := range row.Questions {
		if err := add(question); err != nil {
			return nil, err
		}
		if len(row.Candidates[f]) < 2 || len(row.Candidates[f]) > 64 {
			return nil, fmt.Errorf("mojev: invalid candidate count")
		}
		for _, candidate := range row.Candidates[f] {
			if err := add(candidate); err != nil {
				return nil, err
			}
		}
	}
	result := make([][]float32, len(row.Questions))
	questionStart := len(row.State)
	for f, question := range row.Questions {
		result[f] = make([]float32, len(row.Candidates[f]))
		candidateStart := questionStart + len(question)
		for n, candidate := range row.Candidates[f] {
			length := len(row.State) + len(question) + len(candidate)
			branch := TextBranch{
				IDs: make([]int, 0, length), AbsolutePositions: make([]int, 0, length),
				StateLen: len(row.State), QuestionLen: len(question),
			}
			appendNode := func(ids []int, start int) {
				branch.IDs = append(branch.IDs, ids...)
				for pos := range ids {
					branch.AbsolutePositions = append(branch.AbsolutePositions, start+pos)
				}
			}
			appendNode(row.State, 0)
			appendNode(question, questionStart)
			appendNode(candidate, candidateStart)
			hidden, err := encode(branch)
			if err != nil {
				return nil, fmt.Errorf("mojev: branch %d/%d: %w", f, n, err)
			}
			stateSpan, questionSpan, candidateSpan := make([]bool, length), make([]bool, length), make([]bool, length)
			for i := range stateSpan {
				switch {
				case i < len(row.State):
					stateSpan[i] = true
				case i < len(row.State)+len(question):
					questionSpan[i] = true
				default:
					candidateSpan[i] = true
				}
			}
			logits, err := head.ScoreHidden(hidden, stateSpan, [][]bool{questionSpan}, [][][]bool{{candidateSpan}}, [][]bool{{true}})
			if err != nil {
				return nil, fmt.Errorf("mojev: branch %d/%d head: %w", f, n, err)
			}
			result[f][n] = logits[0][0]
			candidateStart += len(candidate)
		}
		questionStart = candidateStart
	}
	return result, nil
}
