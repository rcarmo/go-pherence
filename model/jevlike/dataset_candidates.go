package jevlike

import (
	"encoding/json"
	"errors"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"
)

// CLINCCandidates defines a labelled candidate-set task, not full-intent
// classification. Gold and OOS are present, remaining candidates favour label
// word overlap, and ties are deterministic. No held-out text is used to fit a
// confusion model; callers retain the generated choices in provenance.
func CLINCCandidates(row json.RawMessage, labels []string, count int, seed int64, oosID int) ([]int, error) {
	if !utf8.Valid(row) || !json.Valid(row) {
		return nil, errors.New("invalid CLINC JSON")
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(row, &fields); err != nil {
		return nil, err
	}
	gold, err := extractCLINCGold(fields, labels)
	if err != nil {
		return nil, err
	}
	if count < 2 || count > 32 || count > len(labels) || gold < 0 || gold >= len(labels) || oosID < 0 || oosID >= len(labels) {
		return nil, errors.New("invalid candidate count or gold/OOS identity")
	}
	text, err := requiredOneJSONString(fields, "text", "utterance", "sentence")
	if err != nil {
		return nil, err
	}
	tokens := map[string]bool{}
	for _, word := range strings.Fields(strings.ToLower(labels[gold])) {
		tokens[word] = true
	}
	type candidate struct {
		id, overlap int
		rank        uint64
	}
	var candidates []candidate
	seen := map[string]bool{}
	for id, label := range labels {
		if err := validateNonBlankUTF8("label", label); err != nil {
			return nil, err
		}
		if seen[label] {
			return nil, errors.New("duplicate intent description")
		}
		seen[label] = true
		if id == gold || id == oosID {
			continue
		}
		overlap := 0
		for _, word := range strings.Fields(strings.ToLower(label)) {
			if tokens[word] {
				overlap++
			}
		}
		candidates = append(candidates, candidate{id, overlap, stableHash64("clinc-candidates-v1", strconv.FormatInt(seed, 10), text, label)})
	}
	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].overlap != candidates[j].overlap {
			return candidates[i].overlap > candidates[j].overlap
		}
		if candidates[i].rank != candidates[j].rank {
			return candidates[i].rank < candidates[j].rank
		}
		return candidates[i].id < candidates[j].id
	})
	ids := []int{gold}
	if oosID != gold {
		ids = append(ids, oosID)
	}
	for _, candidate := range candidates {
		if len(ids) == count {
			break
		}
		ids = append(ids, candidate.id)
	}
	return ids, nil
}
