package mojev

import (
	"reflect"
	"testing"
)

// Called only by the hash-checked released-model test, with its existing scorer.
func testSIMDTreeBranches(t *testing.T, s *SIMDTextScorer) {
	t.Helper()
	testTreeBranches(t, s.ScoreEncoded, func(r EncodedRow) ([][]float32, error) {
		s.mu.Lock()
		defer s.mu.Unlock()
		return ScoreBranchLocalText(r, s.cpu.head, s.encodeBranch)
	})
}
func testNVIDIATreeBranches(t *testing.T, g *NVIDIATextScorer) {
	t.Helper()
	testTreeBranches(t, g.ScoreEncoded, func(r EncodedRow) ([][]float32, error) {
		g.mu.Lock()
		defer g.mu.Unlock()
		return ScoreBranchLocalText(r, g.cpu.head, g.encodeBranch)
	})
}
func testTreeBranches(t *testing.T, score, reference func(EncodedRow) ([][]float32, error)) {
	t.Helper()
	for _, count := range []int{2, 8, 64, 63} {
		r := EncodedRow{State: []int{100, 200, 300}, Questions: [][]int{{400, 500}}, Candidates: make([][][]int, 1)}
		r.Candidates[0] = make([][]int, count)
		for c := range r.Candidates[0] {
			length := 1 + c%4
			if count == 63 {
				length = 6 + c%3
			} // Multiple packed groups in 256-token scratch.
			r.Candidates[0][c] = make([]int, length)
			for i := range r.Candidates[0][c] {
				r.Candidates[0][c][i] = 600 + c*4 + i
			}
		}
		got, err := score(r)
		if err != nil {
			t.Fatal(err)
		}
		want, err := reference(r)
		if err != nil || !reflect.DeepEqual(got, want) {
			t.Fatal("tree vs separate branches", count, err, got, want)
		}
		// Changing candidate order changes packed position and microtile lanes, but
		// not its local RoPE positions or recurrent ancestors.
		for a, b := 0, count-1; a < b; a, b = a+1, b-1 {
			r.Candidates[0][a], r.Candidates[0][b] = r.Candidates[0][b], r.Candidates[0][a]
		}
		reordered, err := score(r)
		if err != nil {
			t.Fatal(err)
		}
		for c, v := range got[0] {
			if v != reordered[0][count-1-c] {
				t.Fatal("tree candidate order leak", count, c)
			}
		}
	}
	// Total tree exceeds the scratch cap; each singleton group still fits.
	r := EncodedRow{State: []int{100}, Questions: [][]int{{200}}, Candidates: [][][]int{{make([]int, 128), make([]int, 128)}}}
	for c := range r.Candidates[0] {
		for i := range r.Candidates[0][c] {
			r.Candidates[0][c][i] = 300 + i + c
		}
	}
	got, err := score(r)
	if err != nil {
		t.Fatal(err)
	}
	want, err := reference(r)
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatal("capacity grouping", err)
	}
}
