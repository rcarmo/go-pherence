package gliner2

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"unicode/utf8"
)

// Entity is the basic native decoder output. All spans are half-open: Byte*
// slice the original Go string, Char* count Unicode code points, and Token*
// index scores.Input.Words.
type Entity struct {
	Label      string  `json:"label"`
	Text       string  `json:"text"`
	Confidence float64 `json:"confidence"`
	ByteStart  int     `json:"byte_start"`
	ByteEnd    int     `json:"byte_end"`
	CharStart  int     `json:"char_start"`
	CharEnd    int     `json:"char_end"`
	TokenStart int     `json:"token_start"`
	TokenEnd   int     `json:"token_end"`
}

// DecodeEntities applies the explicit basic GLiNER2 candidate decoder: sigmoid
// candidate logits, threshold each query independently, collapse exact-boundary
// duplicates, resolve overlaps with the requested policy, and format original
// text slices plus byte/character/token spans. This intentionally does NOT use
// optional abstention/count heads or any adaptive thresholding/count logic.
func DecodeEntities(text string, scores EntityScores, threshold float64, policy string) ([]Entity, error) {
	canonical, err := normalizeOverlapPolicy(policy)
	if err != nil {
		return nil, err
	}
	if err := validateDecodeInput(text, scores, threshold); err != nil {
		return nil, err
	}

	byQuery := make(map[int][]decodedCandidate, len(scores.Input.Labels))
	for candidateIndex := range scores.Candidates.Indices {
		if !scores.Candidates.ValidMask[candidateIndex] {
			continue
		}
		span := scores.Candidates.Indices[candidateIndex]
		start, end := span[0], span[1]
		first, last := scores.Input.Words[start], scores.Input.Words[end-1]
		byteStart, byteEnd := first.ByteStart, last.ByteEnd
		charStart, charEnd := first.Start, last.End
		surface := text[byteStart:byteEnd]
		for queryID, label := range scores.Input.Labels {
			probability := sigmoid(float64(scores.Logits[candidateIndex][queryID]))
			if probability < threshold {
				continue
			}
			byQuery[queryID] = append(byQuery[queryID], decodedCandidate{
				Label:       label,
				Text:        surface,
				Probability: probability,
				ByteStart:   byteStart,
				ByteEnd:     byteEnd,
				CharStart:   charStart,
				CharEnd:     charEnd,
				TokenStart:  start,
				TokenEnd:    end,
				InputIndex:  candidateIndex,
			})
		}
	}

	resolved := make([]decodedCandidate, 0)
	for queryID := range scores.Input.Labels {
		resolved = append(resolved, resolveQueryOverlaps(byQuery[queryID], canonical)...)
	}
	sort.Slice(resolved, func(i, j int) bool { return lessGlobalRank(resolved[i], resolved[j]) })

	entities := make([]Entity, len(resolved))
	for i, candidate := range resolved {
		entities[i] = Entity{
			Label:      candidate.Label,
			Text:       candidate.Text,
			Confidence: candidate.Probability,
			ByteStart:  candidate.ByteStart,
			ByteEnd:    candidate.ByteEnd,
			CharStart:  candidate.CharStart,
			CharEnd:    candidate.CharEnd,
			TokenStart: candidate.TokenStart,
			TokenEnd:   candidate.TokenEnd,
		}
	}
	return entities, nil
}

var overlapPolicyAliases = map[string]string{
	"allow":           "allow",
	"all":             "allow",
	"none":            "allow",
	"nested":          "nested",
	"allow_nested":    "nested",
	"flat":            "disallow",
	"disallow":        "disallow",
	"no_overlap":      "disallow",
	"non_overlapping": "disallow",
	"longest":         "longest",
	"keep_longest":    "longest",
}

func normalizeOverlapPolicy(policy string) (string, error) {
	key := strings.ReplaceAll(strings.ToLower(strings.TrimSpace(policy)), "-", "_")
	canonical, ok := overlapPolicyAliases[key]
	if ok {
		return canonical, nil
	}
	return "", fmt.Errorf("unknown overlap policy %q; expected one of: allow, nested, flat/disallow, longest", policy)
}

func validateDecodeInput(text string, scores EntityScores, threshold float64) error {
	if !utf8.ValidString(text) {
		return fmt.Errorf("decode entities: invalid UTF-8 text")
	}
	if math.IsNaN(threshold) || math.IsInf(threshold, 0) {
		return fmt.Errorf("decode entities: non-finite threshold %v", threshold)
	}
	if len(scores.Input.Labels) == 0 {
		return fmt.Errorf("decode entities: entity labels required")
	}
	seen := make(map[string]bool, len(scores.Input.Labels))
	for i, label := range scores.Input.Labels {
		if strings.TrimSpace(label) == "" {
			return fmt.Errorf("decode entities: empty label at index %d", i)
		}
		if seen[label] {
			return fmt.Errorf("decode entities: duplicate label %q", label)
		}
		seen[label] = true
	}
	if err := validateWords(text, scores.Input.Words); err != nil {
		return err
	}
	candidateCount := len(scores.Candidates.Indices)
	if len(scores.Candidates.ValidMask) != candidateCount {
		return fmt.Errorf("decode entities: valid mask len=%d want=%d", len(scores.Candidates.ValidMask), candidateCount)
	}
	if len(scores.Logits) != candidateCount {
		return fmt.Errorf("decode entities: logits rows=%d want=%d", len(scores.Logits), candidateCount)
	}
	for candidateIndex, span := range scores.Candidates.Indices {
		if len(span) != 2 {
			return fmt.Errorf("decode entities: candidates.indices[%d] len=%d want=2", candidateIndex, len(span))
		}
		if scores.Candidates.ValidMask[candidateIndex] {
			start, end := span[0], span[1]
			if start < 0 || end <= start || end > len(scores.Input.Words) {
				return fmt.Errorf("decode entities: invalid half-open token span [%d,%d) for %d words", start, end, len(scores.Input.Words))
			}
		}
		if len(scores.Logits[candidateIndex]) != len(scores.Input.Labels) {
			return fmt.Errorf("decode entities: logits[%d] len=%d want=%d", candidateIndex, len(scores.Logits[candidateIndex]), len(scores.Input.Labels))
		}
		for queryID, logit := range scores.Logits[candidateIndex] {
			value := float64(logit)
			if math.IsNaN(value) || math.IsInf(value, 0) {
				return fmt.Errorf("decode entities: non-finite logit at candidate=%d query=%d", candidateIndex, queryID)
			}
		}
	}
	return nil
}

func validateWords(text string, words []Word) error {
	prevByteEnd := 0
	prevCharEnd := 0
	for i, word := range words {
		if word.ByteStart < 0 || word.ByteEnd < word.ByteStart || word.ByteEnd > len(text) {
			return fmt.Errorf("decode entities: invalid word[%d] byte span [%d,%d) for text len=%d", i, word.ByteStart, word.ByteEnd, len(text))
		}
		if word.Start < 0 || word.End < word.Start {
			return fmt.Errorf("decode entities: invalid word[%d] char span [%d,%d)", i, word.Start, word.End)
		}
		if word.ByteStart < prevByteEnd {
			return fmt.Errorf("decode entities: word[%d] byte start=%d overlaps previous end=%d", i, word.ByteStart, prevByteEnd)
		}
		if word.Start < prevCharEnd {
			return fmt.Errorf("decode entities: word[%d] char start=%d overlaps previous end=%d", i, word.Start, prevCharEnd)
		}
		gap := text[prevByteEnd:word.ByteStart]
		token := text[word.ByteStart:word.ByteEnd]
		if !utf8.ValidString(gap) {
			return fmt.Errorf("decode entities: word[%d] byte start=%d is not a rune boundary", i, word.ByteStart)
		}
		if !utf8.ValidString(token) {
			return fmt.Errorf("decode entities: word[%d] byte span [%d,%d) is not valid UTF-8", i, word.ByteStart, word.ByteEnd)
		}
		expectedStart := prevCharEnd + utf8.RuneCountInString(gap)
		if word.Start != expectedStart {
			return fmt.Errorf("decode entities: word[%d] char start=%d want=%d", i, word.Start, expectedStart)
		}
		expectedEnd := expectedStart + utf8.RuneCountInString(token)
		if word.End != expectedEnd {
			return fmt.Errorf("decode entities: word[%d] char end=%d want=%d", i, word.End, expectedEnd)
		}
		prevByteEnd = word.ByteEnd
		prevCharEnd = word.End
	}
	return nil
}

func sigmoid(x float64) float64 {
	if x >= 0 {
		return 1 / (1 + math.Exp(-x))
	}
	z := math.Exp(x)
	return z / (1 + z)
}

type decodedCandidate struct {
	Label       string
	Text        string
	Probability float64
	ByteStart   int
	ByteEnd     int
	CharStart   int
	CharEnd     int
	TokenStart  int
	TokenEnd    int
	InputIndex  int
}

func resolveQueryOverlaps(candidates []decodedCandidate, policy string) []decodedCandidate {
	if len(candidates) == 0 {
		return nil
	}
	ranked := append([]decodedCandidate(nil), candidates...)
	sort.Slice(ranked, func(i, j int) bool { return lessQueryRank(ranked[i], ranked[j]) })
	distinct := make([]decodedCandidate, 0, len(ranked))
	seen := make(map[[2]int]bool, len(ranked))
	for _, candidate := range ranked {
		key := [2]int{candidate.TokenStart, candidate.TokenEnd}
		if seen[key] {
			continue
		}
		seen[key] = true
		distinct = append(distinct, candidate)
	}

	switch policy {
	case "allow":
		return distinct
	case "nested":
		kept := make([]decodedCandidate, 0, len(distinct))
		for _, candidate := range distinct {
			crossing := false
			for _, existing := range kept {
				overlaps := candidate.TokenStart < existing.TokenEnd && existing.TokenStart < candidate.TokenEnd
				contains := (candidate.TokenStart <= existing.TokenStart && existing.TokenEnd <= candidate.TokenEnd) ||
					(existing.TokenStart <= candidate.TokenStart && candidate.TokenEnd <= existing.TokenEnd)
				if overlaps && !contains {
					crossing = true
					break
				}
			}
			if !crossing {
				kept = append(kept, candidate)
			}
		}
		return kept
	case "longest":
		kept := make([]decodedCandidate, 0, len(distinct))
		for _, candidate := range distinct {
			contained := false
			for _, other := range distinct {
				if other.TokenStart <= candidate.TokenStart && candidate.TokenEnd <= other.TokenEnd &&
					(other.TokenStart < candidate.TokenStart || candidate.TokenEnd < other.TokenEnd) {
					contained = true
					break
				}
			}
			if !contained {
				kept = append(kept, candidate)
			}
		}
		return kept
	default:
		return resolveFlatOverlaps(distinct)
	}
}

func resolveFlatOverlaps(distinct []decodedCandidate) []decodedCandidate {
	byEnd := append([]decodedCandidate(nil), distinct...)
	sort.Slice(byEnd, func(i, j int) bool {
		if byEnd[i].TokenEnd != byEnd[j].TokenEnd {
			return byEnd[i].TokenEnd < byEnd[j].TokenEnd
		}
		if byEnd[i].TokenStart != byEnd[j].TokenStart {
			return byEnd[i].TokenStart < byEnd[j].TokenStart
		}
		if byEnd[i].Probability != byEnd[j].Probability {
			return byEnd[i].Probability > byEnd[j].Probability
		}
		return byEnd[i].InputIndex < byEnd[j].InputIndex
	})
	ends := make([]int, len(byEnd))
	predecessors := make([]int, len(byEnd))
	for i, candidate := range byEnd {
		ends[i] = candidate.TokenEnd
		predecessors[i] = sort.Search(i, func(j int) bool { return ends[j] > candidate.TokenStart }) - 1
	}
	best := make([]wisState, 1, len(byEnd)+1)
	for i, candidate := range byEnd {
		previous := best[predecessors[i]+1]
		withSelection := make([]int, len(previous.Selection)+1)
		copy(withSelection, previous.Selection)
		withSelection[len(withSelection)-1] = i
		withItem := wisState{Score: previous.Score + candidate.Probability, Selection: withSelection}
		withoutItem := best[i]
		switch {
		case withItem.Score > withoutItem.Score:
			best = append(best, withItem)
		case withItem.Score < withoutItem.Score:
			best = append(best, withoutItem)
		case len(withItem.Selection) > len(withoutItem.Selection):
			best = append(best, withItem)
		case len(withItem.Selection) < len(withoutItem.Selection):
			best = append(best, withoutItem)
		case compareSelectionLex(withItem.Selection, withoutItem.Selection, byEnd) < 0:
			best = append(best, withItem)
		default:
			best = append(best, withoutItem)
		}
	}
	selected := make([]decodedCandidate, len(best[len(best)-1].Selection))
	for i, index := range best[len(best)-1].Selection {
		selected[i] = byEnd[index]
	}
	sort.Slice(selected, func(i, j int) bool { return lessQueryRank(selected[i], selected[j]) })
	return selected
}

type wisState struct {
	Score     float64
	Selection []int
}

func compareSelectionLex(left, right []int, byEnd []decodedCandidate) int {
	leftOrder := append([]int(nil), left...)
	rightOrder := append([]int(nil), right...)
	sort.Slice(leftOrder, func(i, j int) bool { return lessQueryRank(byEnd[leftOrder[i]], byEnd[leftOrder[j]]) })
	sort.Slice(rightOrder, func(i, j int) bool { return lessQueryRank(byEnd[rightOrder[i]], byEnd[rightOrder[j]]) })
	for i := range leftOrder {
		if cmp := compareQueryRank(byEnd[leftOrder[i]], byEnd[rightOrder[i]]); cmp != 0 {
			return cmp
		}
	}
	return 0
}

func compareQueryRank(left, right decodedCandidate) int {
	if left.Probability != right.Probability {
		if left.Probability > right.Probability {
			return -1
		}
		return 1
	}
	if left.TokenStart != right.TokenStart {
		if left.TokenStart < right.TokenStart {
			return -1
		}
		return 1
	}
	if left.TokenEnd != right.TokenEnd {
		if left.TokenEnd < right.TokenEnd {
			return -1
		}
		return 1
	}
	if left.InputIndex < right.InputIndex {
		return -1
	}
	if left.InputIndex > right.InputIndex {
		return 1
	}
	return 0
}

func lessQueryRank(left, right decodedCandidate) bool {
	return compareQueryRank(left, right) < 0
}

func lessGlobalRank(left, right decodedCandidate) bool {
	if left.Probability != right.Probability {
		return left.Probability > right.Probability
	}
	if left.TokenStart != right.TokenStart {
		return left.TokenStart < right.TokenStart
	}
	if left.TokenEnd != right.TokenEnd {
		return left.TokenEnd < right.TokenEnd
	}
	return left.Label < right.Label
}
