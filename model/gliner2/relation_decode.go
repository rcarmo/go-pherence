package gliner2

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"unicode/utf8"

	"golang.org/x/text/cases"
)

// RelationEndpoint is one half-open decoded relation argument span.
// Byte* slice the original Go string; Char* count Unicode code points.
type RelationEndpoint struct {
	Text       string  `json:"text"`
	Confidence float64 `json:"confidence"`
	ByteStart  int     `json:"byte_start"`
	ByteEnd    int     `json:"byte_end"`
	CharStart  int     `json:"char_start"`
	CharEnd    int     `json:"char_end"`
}

// Relation is one decoded typed edge with deduplicated argument mentions.
type Relation struct {
	Type       string           `json:"type"`
	Confidence float64          `json:"confidence"`
	Head       RelationEndpoint `json:"head"`
	Tail       RelationEndpoint `json:"tail"`
}

// DecodeRelations applies the sparse relation decoder used by GLiNER2's public
// boundary engine: temperature-scaled sigmoid scores, thresholding, exact
// half-open word-to-text spans, and per-relation deduplication of contained and
// repeated semantic mention pairs.
func DecodeRelations(text string, scores RelationScores, threshold, temp float64) ([]Relation, error) {
	if err := validateRelationDecodeInput(text, scores, threshold, temp); err != nil {
		return nil, err
	}
	grouped := make(map[string][]decodedRelationEdge)
	for pairIndex, pair := range scores.Pairs.Pairs {
		if len(scores.Pairs.PairMask) != 0 && !scores.Pairs.PairMask[pairIndex] {
			continue
		}
		probability := sigmoid(float64(scores.Logits[pairIndex]) / temp)
		if probability < threshold {
			continue
		}
		head := relationMentionFromTokens(text, scores.Input.Words, pair.HeadStart, pair.HeadEnd)
		tail := relationMentionFromTokens(text, scores.Input.Words, pair.TailStart, pair.TailEnd)
		grouped[pair.RelationType] = append(grouped[pair.RelationType], decodedRelationEdge{
			RelationType: pair.RelationType,
			Score:        probability,
			Head:         head,
			Tail:         tail,
			InputIndex:   pairIndex,
		})
	}
	if len(grouped) == 0 {
		return []Relation{}, nil
	}
	decoded := make([]decodedRelationEdge, 0)
	for _, edges := range grouped {
		decoded = append(decoded, deduplicateRelationEdges(edges)...)
	}
	sort.Slice(decoded, func(i, j int) bool { return lessDecodedRelationEdge(decoded[i], decoded[j]) })
	relations := make([]Relation, len(decoded))
	for i, edge := range decoded {
		relations[i] = Relation{
			Type:       edge.RelationType,
			Confidence: edge.Score,
			Head: RelationEndpoint{
				Text:       edge.Head.Text,
				Confidence: edge.Score,
				ByteStart:  edge.Head.ByteStart,
				ByteEnd:    edge.Head.ByteEnd,
				CharStart:  edge.Head.CharStart,
				CharEnd:    edge.Head.CharEnd,
			},
			Tail: RelationEndpoint{
				Text:       edge.Tail.Text,
				Confidence: edge.Score,
				ByteStart:  edge.Tail.ByteStart,
				ByteEnd:    edge.Tail.ByteEnd,
				CharStart:  edge.Tail.CharStart,
				CharEnd:    edge.Tail.CharEnd,
			},
		}
	}
	return relations, nil
}

func validateRelationDecodeInput(text string, scores RelationScores, threshold, temp float64) error {
	if !utf8.ValidString(text) {
		return fmt.Errorf("decode relations: invalid UTF-8 text")
	}
	if math.IsNaN(threshold) || math.IsInf(threshold, 0) {
		return fmt.Errorf("decode relations: non-finite threshold %v", threshold)
	}
	if math.IsNaN(temp) || math.IsInf(temp, 0) || temp <= 0 {
		return fmt.Errorf("decode relations: temperature must be finite and > 0, got %v", temp)
	}
	if err := validateWords(text, scores.Input.Words); err != nil {
		return err
	}
	pairCount := len(scores.Pairs.Pairs)
	if len(scores.Pairs.PairMask) != 0 && len(scores.Pairs.PairMask) != pairCount {
		return fmt.Errorf("decode relations: pair mask len=%d want=%d", len(scores.Pairs.PairMask), pairCount)
	}
	if len(scores.Logits) != pairCount {
		return fmt.Errorf("decode relations: logits len=%d want=%d", len(scores.Logits), pairCount)
	}
	for pairIndex, pair := range scores.Pairs.Pairs {
		value := float64(scores.Logits[pairIndex])
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return fmt.Errorf("decode relations: non-finite logit at pair=%d", pairIndex)
		}
		if len(scores.Pairs.PairMask) != 0 && !scores.Pairs.PairMask[pairIndex] {
			continue
		}
		if strings.TrimSpace(pair.RelationType) == "" {
			return fmt.Errorf("decode relations: empty relation type at pair=%d", pairIndex)
		}
		if pair.HeadStart < 0 || pair.HeadEnd <= pair.HeadStart || pair.HeadEnd > len(scores.Input.Words) {
			return fmt.Errorf("decode relations: invalid head span [%d,%d) at pair=%d for %d words", pair.HeadStart, pair.HeadEnd, pairIndex, len(scores.Input.Words))
		}
		if pair.TailStart < 0 || pair.TailEnd <= pair.TailStart || pair.TailEnd > len(scores.Input.Words) {
			return fmt.Errorf("decode relations: invalid tail span [%d,%d) at pair=%d for %d words", pair.TailStart, pair.TailEnd, pairIndex, len(scores.Input.Words))
		}
	}
	return nil
}

type relationCoordinates struct {
	Start int
	End   int
}

type decodedRelationMention struct {
	Text               string
	ByteStart, ByteEnd int
	CharStart, CharEnd int
}

type decodedRelationEdge struct {
	RelationType string
	Score        float64
	Head         decodedRelationMention
	Tail         decodedRelationMention
	InputIndex   int
}

func relationMentionFromTokens(text string, words []Word, start, end int) decodedRelationMention {
	first, last := words[start], words[end-1]
	return decodedRelationMention{
		Text:      text[first.ByteStart:last.ByteEnd],
		ByteStart: first.ByteStart,
		ByteEnd:   last.ByteEnd,
		CharStart: first.Start,
		CharEnd:   last.End,
	}
}

func deduplicateRelationEdges(edges []decodedRelationEdge) []decodedRelationEdge {
	if len(edges) < 2 {
		out := append([]decodedRelationEdge(nil), edges...)
		sort.Slice(out, func(i, j int) bool { return lessDecodedRelationEdge(out[i], out[j]) })
		return out
	}
	headCanonical := canonicalRelationMentions(edges, func(edge decodedRelationEdge) decodedRelationMention { return edge.Head })
	tailCanonical := canonicalRelationMentions(edges, func(edge decodedRelationEdge) decodedRelationMention { return edge.Tail })

	type endpointKey struct {
		Head relationCoordinates
		Tail relationCoordinates
	}
	exactIndex := make(map[endpointKey]int, len(edges))
	exact := make([]decodedRelationEdge, 0, len(edges))
	for _, edge := range edges {
		head := headCanonical[relationCoordinates{Start: edge.Head.CharStart, End: edge.Head.CharEnd}]
		tail := tailCanonical[relationCoordinates{Start: edge.Tail.CharStart, End: edge.Tail.CharEnd}]
		normalized := edge
		normalized.Head = head
		normalized.Tail = tail
		key := endpointKey{
			Head: relationCoordinates{Start: head.CharStart, End: head.CharEnd},
			Tail: relationCoordinates{Start: tail.CharStart, End: tail.CharEnd},
		}
		if index, ok := exactIndex[key]; ok {
			if normalized.Score > exact[index].Score {
				exact[index] = normalized
			}
			continue
		}
		exactIndex[key] = len(exact)
		exact = append(exact, normalized)
	}

	semanticIndex := make(map[[2]string]int, len(exact))
	semantic := make([]decodedRelationEdge, 0, len(exact))
	for _, edge := range exact {
		key := [2]string{semanticText(edge.Head.Text), semanticText(edge.Tail.Text)}
		if index, ok := semanticIndex[key]; ok {
			if lessSemanticRelationRank(edge, semantic[index]) {
				semantic[index] = edge
			}
			continue
		}
		semanticIndex[key] = len(semantic)
		semantic = append(semantic, edge)
	}

	headSets := make([]tokenSet, len(semantic))
	tailSets := make([]tokenSet, len(semantic))
	for i, edge := range semantic {
		headSets[i] = semanticTokenSet(edge.Head.Text)
		tailSets[i] = semanticTokenSet(edge.Tail.Text)
	}
	kept := make([]decodedRelationEdge, 0, len(semantic))
	for i, edge := range semantic {
		dominated := false
		for j := range semantic {
			if i == j {
				continue
			}
			if strictTokenSubset(headSets[i], headSets[j]) && equalTokenSet(tailSets[i], tailSets[j]) {
				dominated = true
				break
			}
			if strictTokenSubset(tailSets[i], tailSets[j]) && equalTokenSet(headSets[i], headSets[j]) {
				dominated = true
				break
			}
		}
		if !dominated {
			kept = append(kept, edge)
		}
	}
	sort.Slice(kept, func(i, j int) bool { return lessDecodedRelationEdge(kept[i], kept[j]) })
	return kept
}

func canonicalRelationMentions(edges []decodedRelationEdge, selectMention func(decodedRelationEdge) decodedRelationMention) map[relationCoordinates]decodedRelationMention {
	mentions := make([]decodedRelationMention, 0, len(edges))
	seen := make(map[relationCoordinates]bool, len(edges))
	for _, edge := range edges {
		mention := selectMention(edge)
		key := relationCoordinates{Start: mention.CharStart, End: mention.CharEnd}
		if seen[key] {
			continue
		}
		seen[key] = true
		mentions = append(mentions, mention)
	}
	canonical := make(map[relationCoordinates]decodedRelationMention, len(mentions))
	for _, mention := range mentions {
		best := mention
		for _, candidate := range mentions {
			if candidate.CharStart <= mention.CharStart && candidate.CharEnd >= mention.CharEnd && betterCanonicalMention(candidate, best) {
				best = candidate
			}
		}
		canonical[relationCoordinates{Start: mention.CharStart, End: mention.CharEnd}] = best
	}
	return canonical
}

func betterCanonicalMention(candidate, current decodedRelationMention) bool {
	candidateWidth := candidate.CharEnd - candidate.CharStart
	currentWidth := current.CharEnd - current.CharStart
	if candidateWidth != currentWidth {
		return candidateWidth > currentWidth
	}
	if candidate.CharStart != current.CharStart {
		return candidate.CharStart < current.CharStart
	}
	if candidate.CharEnd != current.CharEnd {
		return candidate.CharEnd > current.CharEnd
	}
	if candidate.ByteStart != current.ByteStart {
		return candidate.ByteStart < current.ByteStart
	}
	if candidate.ByteEnd != current.ByteEnd {
		return candidate.ByteEnd > current.ByteEnd
	}
	return false
}

var relationCaseFold = cases.Fold()

func semanticText(value string) string {
	return strings.Join(strings.Fields(relationCaseFold.String(value)), " ")
}

func lessSemanticRelationRank(left, right decodedRelationEdge) bool {
	leftDistance := relationDistance(left)
	rightDistance := relationDistance(right)
	if leftDistance != rightDistance {
		return leftDistance < rightDistance
	}
	if left.Score != right.Score {
		return left.Score > right.Score
	}
	if left.Head.CharStart != right.Head.CharStart {
		return left.Head.CharStart < right.Head.CharStart
	}
	if left.Tail.CharStart != right.Tail.CharStart {
		return left.Tail.CharStart < right.Tail.CharStart
	}
	if left.Head.CharEnd != right.Head.CharEnd {
		return left.Head.CharEnd < right.Head.CharEnd
	}
	if left.Tail.CharEnd != right.Tail.CharEnd {
		return left.Tail.CharEnd < right.Tail.CharEnd
	}
	return left.InputIndex < right.InputIndex
}

func relationDistance(edge decodedRelationEdge) int {
	distance := max(edge.Head.CharStart-edge.Tail.CharEnd, edge.Tail.CharStart-edge.Head.CharEnd)
	if distance < 0 {
		return 0
	}
	return distance
}

type tokenSet map[string]struct{}

func semanticTokenSet(value string) tokenSet {
	out := make(tokenSet)
	for _, token := range strings.Fields(semanticText(value)) {
		out[token] = struct{}{}
	}
	return out
}

func equalTokenSet(left, right tokenSet) bool {
	if len(left) != len(right) {
		return false
	}
	for token := range left {
		if _, ok := right[token]; !ok {
			return false
		}
	}
	return true
}

func strictTokenSubset(left, right tokenSet) bool {
	if len(left) >= len(right) {
		return false
	}
	for token := range left {
		if _, ok := right[token]; !ok {
			return false
		}
	}
	return true
}

func lessDecodedRelationEdge(left, right decodedRelationEdge) bool {
	if left.Head.CharStart != right.Head.CharStart {
		return left.Head.CharStart < right.Head.CharStart
	}
	if left.Tail.CharStart != right.Tail.CharStart {
		return left.Tail.CharStart < right.Tail.CharStart
	}
	if left.Score != right.Score {
		return left.Score > right.Score
	}
	if left.RelationType != right.RelationType {
		return left.RelationType < right.RelationType
	}
	if left.Head.CharEnd != right.Head.CharEnd {
		return left.Head.CharEnd < right.Head.CharEnd
	}
	if left.Tail.CharEnd != right.Tail.CharEnd {
		return left.Tail.CharEnd < right.Tail.CharEnd
	}
	return left.InputIndex < right.InputIndex
}
