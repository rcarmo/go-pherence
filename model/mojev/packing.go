package mojev

import "fmt"

// EncodedRow contains pre-tokenised text segments in original question and
// candidate order. Tokenisation and prompt rendering are separate gates.
type EncodedRow struct {
	State      []int
	Questions  [][]int
	Candidates [][][]int
}

// PackedRows holds request-owned, batch-padded token IDs and span indicators.
// Unused candidate slots have empty spans and false OptionMask entries.
type PackedRows struct {
	IDs        [][]int
	PackedMask [][]bool
	State      [][]bool
	Questions  [][][]bool
	Candidates [][][][]bool
	OptionMask [][][]bool
}

// PackEncodedRows mirrors the text-only state/question/candidate sequence layout
// for a batch with already-tokenised segments. It does not render prompts,
// tokenize text, expand images, or run the encoder.
func PackEncodedRows(rows []EncodedRow, padID int) (*PackedRows, error) {
	if len(rows) == 0 || len(rows) > 16 || padID < 0 {
		return nil, fmt.Errorf("mojev: invalid packing batch or pad ID")
	}
	fields := len(rows[0].Questions)
	if fields == 0 || fields > 256 {
		return nil, fmt.Errorf("mojev: invalid packing question count")
	}
	width, maxLength := 0, 0
	for _, row := range rows {
		if len(row.Questions) != fields || len(row.Candidates) != fields || len(row.State) == 0 {
			return nil, fmt.Errorf("mojev: inconsistent packing geometry")
		}
		length := len(row.State)
		for f, q := range row.Questions {
			if len(q) == 0 || len(row.Candidates[f]) < 2 || len(row.Candidates[f]) > 64 {
				return nil, fmt.Errorf("mojev: empty question or invalid candidate count")
			}
			if len(row.Candidates[f]) > width {
				width = len(row.Candidates[f])
			}
			length += len(q)
			for _, candidate := range row.Candidates[f] {
				if len(candidate) == 0 {
					return nil, fmt.Errorf("mojev: empty candidate segment")
				}
				length += len(candidate)
			}
		}
		if length > 4096 {
			return nil, fmt.Errorf("mojev: packed row exceeds reference limit")
		}
		for _, id := range row.State {
			if id < 0 {
				return nil, fmt.Errorf("mojev: negative token ID")
			}
		}
		for _, q := range row.Questions {
			for _, id := range q {
				if id < 0 {
					return nil, fmt.Errorf("mojev: negative token ID")
				}
			}
		}
		for _, group := range row.Candidates {
			for _, c := range group {
				for _, id := range c {
					if id < 0 {
						return nil, fmt.Errorf("mojev: negative token ID")
					}
				}
			}
		}
		if length > maxLength {
			maxLength = length
		}
	}
	out := &PackedRows{
		IDs: make([][]int, len(rows)), PackedMask: make([][]bool, len(rows)),
		State: make([][]bool, len(rows)), Questions: make([][][]bool, len(rows)),
		Candidates: make([][][][]bool, len(rows)), OptionMask: make([][][]bool, len(rows)),
	}
	for r, row := range rows {
		ids := make([]int, maxLength)
		for i := range ids {
			ids[i] = padID
		}
		packed := make([]bool, maxLength)
		state := make([]bool, maxLength)
		questions := make([][]bool, fields)
		candidates := make([][][]bool, fields)
		optionMask := make([][]bool, fields)
		cursor := 0
		appendSpan := func(tokens []int, span []bool) {
			copy(ids[cursor:], tokens)
			for i := range tokens {
				packed[cursor+i] = true
				span[cursor+i] = true
			}
			cursor += len(tokens)
		}
		appendSpan(row.State, state)
		for f, q := range row.Questions {
			questions[f] = make([]bool, maxLength)
			appendSpan(q, questions[f])
			candidates[f] = make([][]bool, width)
			optionMask[f] = make([]bool, width)
			for n := 0; n < width; n++ {
				candidates[f][n] = make([]bool, maxLength)
				if n < len(row.Candidates[f]) {
					appendSpan(row.Candidates[f][n], candidates[f][n])
					optionMask[f][n] = true
				}
			}
		}
		out.IDs[r], out.PackedMask[r], out.State[r] = ids, packed, state
		out.Questions[r], out.Candidates[r], out.OptionMask[r] = questions, candidates, optionMask
	}
	return out, nil
}
