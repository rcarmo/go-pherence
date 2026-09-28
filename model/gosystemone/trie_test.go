package gosystemone

import (
	"math"
	"strings"
	"testing"
)

type wordTokenizer struct{}

func (wordTokenizer) Encode(text string) []int {
	ids := map[string]int{"key": 1, "a": 2, "b": 3, "c": 4, "d": 5, "e": 6, "\n": 7}
	var out []int
	for _, part := range strings.Fields(strings.ReplaceAll(text, "\n", " \n ")) {
		for _, r := range part {
			out = append(out, ids[string(r)])
		}
	}
	return out
}

func testInput() FieldInput {
	return FieldInput{Suffix: "key", Candidates: []Candidate{
		{ID: "x:000", Encoded: "ab"},
		{ID: "x:001", Encoded: "ac"},
		{ID: "x:002", Encoded: "de"},
	}}
}

func TestCompileFieldBuildsDeterministicDivergenceTrie(t *testing.T) {
	field, err := CompileField(wordTokenizer{}, testInput(), ModeTree, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !field.Tree || len(field.Nodes) != 2 {
		t.Fatalf("tree=%v nodes=%+v", field.Tree, field.Nodes)
	}
	if got, want := field.Nodes[0].Options, []int{2, 5}; !sameTokens(got, want) {
		t.Fatalf("root options=%v want %v", got, want)
	}
	if got, want := field.Nodes[1].Prefix, []int{2}; !sameTokens(got, want) {
		t.Fatalf("second prefix=%v want %v", got, want)
	}
	if got, want := field.Nodes[1].Options, []int{3, 4}; !sameTokens(got, want) {
		t.Fatalf("second options=%v want %v", got, want)
	}
	if field.Rows != len(field.Suffix)*2+1 {
		t.Fatalf("rows=%d suffix=%d", field.Rows, len(field.Suffix))
	}
}

func TestFinishTreeExactConstrainedDistribution(t *testing.T) {
	field, err := CompileField(wordTokenizer{}, testInput(), ModeTree, 0)
	if err != nil {
		t.Fatal(err)
	}
	// Root strongly prefers the a* subtree. Within it, c beats b.
	winner, probs, err := FinishTree(field, [][]float32{{2, 0}, {0, 1}})
	if err != nil {
		t.Fatal(err)
	}
	if winner != 1 {
		t.Fatalf("winner=%d probabilities=%v", winner, probs)
	}
	if math.Abs(probs[0]-0.2368828) > 1e-6 || math.Abs(probs[1]-0.6439143) > 1e-6 || math.Abs(probs[2]-0.1192029) > 1e-6 {
		t.Fatalf("probabilities=%v", probs)
	}
	sum := 0.0
	for _, probability := range probs {
		sum += probability
	}
	if math.Abs(sum-1) > 1e-12 {
		t.Fatalf("probability sum=%g", sum)
	}
}

func TestCompileFieldAutoThresholdAndCollision(t *testing.T) {
	field, err := CompileField(wordTokenizer{}, testInput(), ModeAuto, 2)
	if err != nil {
		t.Fatal(err)
	}
	if field.Tree {
		t.Fatal("auto mode ignored tree_max")
	}
	collision := FieldInput{Suffix: "key", Candidates: []Candidate{{ID: "a", Encoded: "a"}, {ID: "ab", Encoded: "ab"}}}
	if _, err := CompileField(wordTokenizer{}, collision, ModeTree, 0); err == nil || !strings.Contains(err.Error(), "colliding") {
		t.Fatalf("collision error=%v", err)
	}
}
