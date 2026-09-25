package mojev

import (
	"reflect"
	"testing"
)

// Called only by the hash-checked released-model test, with its existing scorer.
func testSIMDTreeBranches(t *testing.T, s *SIMDTextScorer) {
	t.Helper()
	for _, count := range []int{2, 8, 64} {
		r := EncodedRow{State: []int{100, 200, 300}, Questions: [][]int{{400, 500}}, Candidates: make([][][]int, 1)}
		r.Candidates[0] = make([][]int, count)
		for c := range r.Candidates[0] {
			r.Candidates[0][c] = make([]int, 1+c%4)
			for i := range r.Candidates[0][c] {
				r.Candidates[0][c][i] = 600 + c*4 + i
			}
		}
		got, err := s.ScoreEncoded(r)
		if err != nil {
			t.Fatal(err)
		}
		s.mu.Lock()
		want, err := ScoreBranchLocalText(r, s.cpu.head, s.encodeBranch)
		s.mu.Unlock()
		if err != nil || !reflect.DeepEqual(got, want) {
			t.Fatal("tree vs separate branches", count, err, got, want)
		}
		// Changing candidate order changes packed position and microtile lanes, but
		// not its local RoPE positions or recurrent ancestors.
		for a, b := 0, count-1; a < b; a, b = a+1, b-1 {
			r.Candidates[0][a], r.Candidates[0][b] = r.Candidates[0][b], r.Candidates[0][a]
		}
		reordered, err := s.ScoreEncoded(r)
		if err != nil {
			t.Fatal(err)
		}
		for c, v := range got[0] {
			if v != reordered[0][count-1-c] {
				t.Fatal("tree candidate order leak", count, c)
			}
		}
	}
	// Total tree exceeds the scratch cap; each individual path still fits.
	r := EncodedRow{State: []int{100}, Questions: [][]int{{200}}, Candidates: [][][]int{{make([]int, 128), make([]int, 128)}}}
	for c := range r.Candidates[0] {
		for i := range r.Candidates[0][c] {
			r.Candidates[0][c][i] = 300 + i + c
		}
	}
	got, err := s.ScoreEncoded(r)
	if err != nil {
		t.Fatal(err)
	}
	s.mu.Lock()
	want, err := ScoreBranchLocalText(r, s.cpu.head, s.encodeBranch)
	s.mu.Unlock()
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatal("capacity fallback", err)
	}
}
