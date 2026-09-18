package gliner2

import (
	"math"
	"strings"
	"testing"
	"unicode/utf8"
)

type relationRow struct {
	RelationType   string
	HeadTokens     []string
	HeadOccurrence int
	TailTokens     []string
	TailOccurrence int
	Prob           float64
}

func TestDecodeRelationsContainingMentionsCollapseToWidestEndpoint(t *testing.T) {
	text := "Ada Lovelace lived in London."
	scores := makeRelationScores(t, text, []relationRow{
		{RelationType: "lives_in", HeadTokens: []string{"ada"}, HeadOccurrence: 0, TailTokens: []string{"london"}, TailOccurrence: 0, Prob: 0.91},
		{RelationType: "lives_in", HeadTokens: []string{"ada", "lovelace"}, HeadOccurrence: 0, TailTokens: []string{"london"}, TailOccurrence: 0, Prob: 0.80},
	})

	got, err := DecodeRelations(text, scores, 0.5, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 {
		t.Fatalf("len=%d want=1", len(got))
	}
	relation := got[0]
	if relation.Type != "lives_in" {
		t.Fatalf("type=%q want lives_in", relation.Type)
	}
	if relation.Head.Text != "Ada Lovelace" {
		t.Fatalf("head=%q want %q", relation.Head.Text, "Ada Lovelace")
	}
	if relation.Tail.Text != "London" {
		t.Fatalf("tail=%q want %q", relation.Tail.Text, "London")
	}
	if math.Abs(relation.Confidence-0.91) > 1e-6 {
		t.Fatalf("confidence=%g want 0.91", relation.Confidence)
	}
	if relation.Head.Confidence != relation.Confidence || relation.Tail.Confidence != relation.Confidence {
		t.Fatalf("endpoint confidences=%g/%g want %g", relation.Head.Confidence, relation.Tail.Confidence, relation.Confidence)
	}
	assertEndpointMatchesSubstring(t, text, relation.Head, "Ada Lovelace", 0)
	assertEndpointMatchesSubstring(t, text, relation.Tail, "London", 0)
}

func TestDecodeRelationsRepeatedMentionsPreferClosestSemanticPair(t *testing.T) {
	text := "Alice met Bob. Alice Bob."
	scores := makeRelationScores(t, text, []relationRow{
		{RelationType: "knows", HeadTokens: []string{"alice"}, HeadOccurrence: 0, TailTokens: []string{"bob"}, TailOccurrence: 0, Prob: 0.95},
		{RelationType: "knows", HeadTokens: []string{"alice"}, HeadOccurrence: 1, TailTokens: []string{"bob"}, TailOccurrence: 1, Prob: 0.70},
	})

	got, err := DecodeRelations(text, scores, 0.5, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 {
		t.Fatalf("len=%d want=1", len(got))
	}
	relation := got[0]
	if relation.Head.ByteStart != strings.LastIndex(text, "Alice") {
		t.Fatalf("head byte_start=%d want %d", relation.Head.ByteStart, strings.LastIndex(text, "Alice"))
	}
	if relation.Tail.ByteStart != strings.LastIndex(text, "Bob") {
		t.Fatalf("tail byte_start=%d want %d", relation.Tail.ByteStart, strings.LastIndex(text, "Bob"))
	}
	if math.Abs(relation.Confidence-0.70) > 1e-6 {
		t.Fatalf("confidence=%g want 0.70", relation.Confidence)
	}
}

func TestDecodeRelationsStrictSubsetDominatesButEqualTokenSetsRemain(t *testing.T) {
	text := "York USA. New York USA. Francisco San California. San Francisco California."
	scores := makeRelationScores(t, text, []relationRow{
		{RelationType: "located_in", HeadTokens: []string{"york"}, HeadOccurrence: 0, TailTokens: []string{"usa"}, TailOccurrence: 0, Prob: 0.95},
		{RelationType: "located_in", HeadTokens: []string{"new", "york"}, HeadOccurrence: 0, TailTokens: []string{"usa"}, TailOccurrence: 1, Prob: 0.80},
		{RelationType: "located_in", HeadTokens: []string{"francisco", "san"}, HeadOccurrence: 0, TailTokens: []string{"california"}, TailOccurrence: 0, Prob: 0.75},
		{RelationType: "located_in", HeadTokens: []string{"san", "francisco"}, HeadOccurrence: 0, TailTokens: []string{"california"}, TailOccurrence: 1, Prob: 0.70},
	})

	got, err := DecodeRelations(text, scores, 0.5, 1)
	if err != nil {
		t.Fatal(err)
	}
	assertRelationTriples(t, got, [][3]string{
		{"located_in", "New York", "USA"},
		{"located_in", "Francisco San", "California"},
		{"located_in", "San Francisco", "California"},
	})
}

func TestDecodeRelationsUnicodeCasefoldDeduplicatesSemanticPairs(t *testing.T) {
	text := "Straße Berlin. STRASSE Berlin."
	scores := makeRelationScores(t, text, []relationRow{
		{RelationType: "near", HeadTokens: []string{"straße"}, HeadOccurrence: 0, TailTokens: []string{"berlin"}, TailOccurrence: 0, Prob: 0.90},
		{RelationType: "near", HeadTokens: []string{"strasse"}, HeadOccurrence: 0, TailTokens: []string{"berlin"}, TailOccurrence: 1, Prob: 0.80},
	})

	got, err := DecodeRelations(text, scores, 0.5, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 {
		t.Fatalf("len=%d want=1", len(got))
	}
	relation := got[0]
	if relation.Head.Text != "Straße" {
		t.Fatalf("head=%q want %q", relation.Head.Text, "Straße")
	}
	assertEndpointMatchesSubstring(t, text, relation.Head, "Straße", 0)
	if relation.Head.CharEnd-relation.Head.CharStart != utf8.RuneCountInString("Straße") {
		t.Fatalf("head char width=%d want %d", relation.Head.CharEnd-relation.Head.CharStart, utf8.RuneCountInString("Straße"))
	}
	if relation.Head.ByteEnd-relation.Head.ByteStart != len("Straße") {
		t.Fatalf("head byte width=%d want %d", relation.Head.ByteEnd-relation.Head.ByteStart, len("Straße"))
	}
}

func TestDecodeRelationsAppliesTemperature(t *testing.T) {
	text := "Ada London"
	scores := makeRelationScores(t, text, []relationRow{{RelationType: "lives_in", HeadTokens: []string{"ada"}, HeadOccurrence: 0, TailTokens: []string{"london"}, TailOccurrence: 0, Prob: 0.80}})

	warm, err := DecodeRelations(text, scores, 0.7, 1)
	if err != nil {
		t.Fatal(err)
	}
	cool, err := DecodeRelations(text, scores, 0.7, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(warm) != 1 {
		t.Fatalf("warm len=%d want=1", len(warm))
	}
	if len(cool) != 0 {
		t.Fatalf("cool len=%d want=0", len(cool))
	}
}

func TestDecodeRelationsRejectsMalformedInput(t *testing.T) {
	text := "Ada London"
	base := makeRelationScores(t, text, []relationRow{{RelationType: "lives_in", HeadTokens: []string{"ada"}, HeadOccurrence: 0, TailTokens: []string{"london"}, TailOccurrence: 0, Prob: 0.80}})

	cases := []struct {
		name string
		edit func(RelationScores) RelationScores
		temp float64
		want string
	}{
		{
			name: "pair mask len",
			edit: func(scores RelationScores) RelationScores {
				scores.Pairs.PairMask = []bool{true, true}
				return scores
			},
			temp: 1,
			want: "pair mask",
		},
		{
			name: "non-finite logit",
			edit: func(scores RelationScores) RelationScores {
				scores.Logits[0] = float32(math.Inf(1))
				return scores
			},
			temp: 1,
			want: "non-finite logit",
		},
		{
			name: "invalid head span",
			edit: func(scores RelationScores) RelationScores {
				scores.Pairs.Pairs[0].HeadEnd = len(scores.Input.Words) + 1
				return scores
			},
			temp: 1,
			want: "invalid head span",
		},
		{
			name: "invalid temperature",
			edit: func(scores RelationScores) RelationScores {
				return scores
			},
			temp: 0,
			want: "temperature",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := DecodeRelations(text, tc.edit(cloneRelationScores(base)), 0.5, tc.temp)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err=%v want substring %q", err, tc.want)
			}
		})
	}
}

func makeRelationScores(t *testing.T, text string, rows []relationRow) RelationScores {
	t.Helper()
	words, err := SplitWords(text)
	if err != nil {
		t.Fatal(err)
	}
	pairs := make([]RelationPair, len(rows))
	mask := make([]bool, len(rows))
	logits := make([]float32, len(rows))
	for i, row := range rows {
		hs, he := findTokenSpan(t, words, row.HeadOccurrence, row.HeadTokens...)
		ts, te := findTokenSpan(t, words, row.TailOccurrence, row.TailTokens...)
		pairs[i] = RelationPair{
			RelationIndex: 0,
			RelationType:  row.RelationType,
			HeadStart:     hs,
			HeadEnd:       he,
			TailStart:     ts,
			TailEnd:       te,
		}
		mask[i] = true
		logits[i] = float32(logit(row.Prob))
	}
	return RelationScores{
		Input:  EntityInput{Words: words},
		Pairs:  RelationPairProposals{Pairs: pairs, PairMask: mask},
		Logits: logits,
	}
}

func findTokenSpan(t *testing.T, words []Word, occurrence int, tokens ...string) (int, int) {
	t.Helper()
	seen := 0
	for start := 0; start+len(tokens) <= len(words); start++ {
		matched := true
		for i, token := range tokens {
			if words[start+i].Text != token {
				matched = false
				break
			}
		}
		if !matched {
			continue
		}
		if seen == occurrence {
			return start, start + len(tokens)
		}
		seen++
	}
	t.Fatalf("tokens %v occurrence %d not found in %v", tokens, occurrence, wordTexts(words))
	return 0, 0
}

func wordTexts(words []Word) []string {
	out := make([]string, len(words))
	for i, word := range words {
		out[i] = word.Text
	}
	return out
}

func cloneRelationScores(in RelationScores) RelationScores {
	out := in
	out.Input = in.Input
	out.Input.Words = append([]Word(nil), in.Input.Words...)
	out.Pairs = RelationPairProposals{
		Pairs:    append([]RelationPair(nil), in.Pairs.Pairs...),
		PairMask: append([]bool(nil), in.Pairs.PairMask...),
	}
	out.Logits = append([]float32(nil), in.Logits...)
	return out
}

func assertRelationTriples(t *testing.T, got []Relation, want [][3]string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("len=%d want=%d", len(got), len(want))
	}
	for i, relation := range got {
		triple := [3]string{relation.Type, relation.Head.Text, relation.Tail.Text}
		if triple != want[i] {
			t.Fatalf("triple[%d]=%v want %v", i, triple, want[i])
		}
	}
}

func assertEndpointMatchesSubstring(t *testing.T, text string, endpoint RelationEndpoint, substring string, occurrence int) {
	t.Helper()
	byteStart := nthIndex(text, substring, occurrence)
	if byteStart < 0 {
		t.Fatalf("substring %q occurrence %d not found in %q", substring, occurrence, text)
	}
	byteEnd := byteStart + len(substring)
	charStart := utf8.RuneCountInString(text[:byteStart])
	charEnd := utf8.RuneCountInString(text[:byteEnd])
	if endpoint.Text != substring {
		t.Fatalf("text=%q want %q", endpoint.Text, substring)
	}
	if endpoint.ByteStart != byteStart || endpoint.ByteEnd != byteEnd {
		t.Fatalf("byte span=[%d,%d) want [%d,%d)", endpoint.ByteStart, endpoint.ByteEnd, byteStart, byteEnd)
	}
	if endpoint.CharStart != charStart || endpoint.CharEnd != charEnd {
		t.Fatalf("char span=[%d,%d) want [%d,%d)", endpoint.CharStart, endpoint.CharEnd, charStart, charEnd)
	}
}

func nthIndex(text, substring string, occurrence int) int {
	if occurrence < 0 {
		return -1
	}
	searchStart := 0
	for i := 0; ; i++ {
		offset := strings.Index(text[searchStart:], substring)
		if offset < 0 {
			return -1
		}
		at := searchStart + offset
		if i == occurrence {
			return at
		}
		searchStart = at + len(substring)
	}
}
