package jevlike

import (
	"math"
	"testing"
)

func permutations(n int) [][]int {
	var result [][]int
	var visit func([]int, []bool)
	visit = func(p []int, used []bool) {
		if len(p) == n {
			result = append(result, append([]int(nil), p...))
			return
		}
		for i := range n {
			if !used[i] {
				used[i] = true
				visit(append(p, i), used)
				used[i] = false
			}
		}
	}
	visit(nil, make([]bool, n))
	return result
}

func TestFrozenHeadPermutationEquivarianceByStableID(t *testing.T) {
	for _, dtype := range []string{"f32", "f16"} {
		contract := cacheTestContract(dtype)
		cache, err := OpenFeatureCache(t.TempDir(), contract, 1<<20, cacheTokenizer, cacheSynthetic)
		if err != nil {
			t.Fatal(err)
		}
		h, _ := NewAttentionHead(4, 3)
		m := &FrozenScorer{Config: Config{Width: 4, Rank: 3, ContextTokens: 8, OptionTokens: 4}, Reference: cache.FeatureReference(), Head: *h, Encoder: cache}
		if err = InitializeFrozenScorer(m, 7); err != nil {
			t.Fatal(err)
		}
		for n := 2; n <= 4; n++ {
			opts := []string{"a", "bc", "def", "ghij"}[:n]
			original := ChoiceExample{Context: "abcdef", Options: opts, Label: 1}
			base, err := m.Forward([]ChoiceExample{original}, false)
			if err != nil {
				t.Fatal(err)
			}
			// Include mixed choice counts/context lengths to exercise padding masks.
			for _, perm := range permutations(n) {
				ex := ChoiceExample{Context: original.Context, Options: make([]string, n)}
				for j, i := range perm {
					ex.Options[j] = opts[i]
					if i == original.Label {
						ex.Label = j
					}
				}
				got, err := m.Forward([]ChoiceExample{ex, {Context: "a", Options: []string{"a", "b"}, Label: 0}}, false)
				if err != nil {
					t.Fatal(err)
				}
				for j, i := range perm {
					if got[0][j] != base[0][i] {
						t.Fatalf("%s permutation %v stable candidate %q logit drift %g", dtype, perm, opts[i], got[0][j]-base[0][i])
					}
				}
				a, _ := DecisionMetricsAtTemperature(base, []int{original.Label}, 1)
				b, _ := DecisionMetricsAtTemperature(got[:1], []int{ex.Label}, 1)
				if math.Abs(a.NLL-b.NLL) > 1e-12 || math.Abs(a.Brier-b.Brier) > 1e-12 {
					t.Fatal("label/probability permutation mismatch")
				}
			}
		}
		cache.Close()
	}
}

func TestSelfContainedCandidateContract(t *testing.T) {
	r := directTestRequest()
	r.CandidateContract = SelfContainedCandidatesV1
	r.Candidates = []ChoiceCandidate{{ID: "yes", Text: "Ada lives in Lisbon."}, {ID: "no", Text: "Ada lives in Paris."}}
	if err := ValidateCandidateContract(r); err != nil {
		t.Fatal(err)
	}
	for _, text := range []string{"A", "option B", "yes", "No.", "all of the above", "that"} {
		copy := r
		copy.Candidates = append([]ChoiceCandidate(nil), r.Candidates...)
		copy.Candidates[0].Text = text
		if err := ValidateCandidateContract(copy); err == nil {
			t.Fatal("context-dependent candidate accepted", text)
		}
	}
	r.CandidateContract = ""
	if err := ValidateCandidateContract(r); err == nil {
		t.Fatal("undeclared contract")
	}
	r.CandidateContract = BenchmarkFragmentsV1
	r.Candidates[0].Text = "yes"
	if err := ValidateCandidateContract(r); err != nil {
		t.Fatal("explicit benchmark limitation rejected", err)
	}
}
