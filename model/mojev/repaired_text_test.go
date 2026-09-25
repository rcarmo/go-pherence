package mojev

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"math"
	"os"
	"reflect"
	"sync"
	"testing"
)

func repairedTextRow() EncodedRow {
	return EncodedRow{State: []int{10, 15, 21, 22}, Questions: [][]int{{31, 32}, {51, 52}},
		Candidates: [][][]int{{{41}, {42}}, {{61}, {62}}}}
}

func TestRepairedTextFixture(t *testing.T) {
	for _, item := range []struct{ path, want string }{
		{"../../scripts/mojev_oracle_repaired_text_isolation.py", "7fe9998def69d1b14fede716d750fc527e84178ef5eeacaa1d3691edbf45ed30"},
		{"testdata/repaired_text_isolation.json", "873ad0808a9b0578a4143824cb290353fb70a6f547998dd85f780dafa41ca9d3"},
	} {
		data, err := os.ReadFile(item.path)
		if err != nil {
			t.Fatal(err)
		}
		sum := sha256.Sum256(data)
		if hex.EncodeToString(sum[:]) != item.want {
			t.Fatalf("%s hash mismatch", item.path)
		}
	}
	data, err := os.ReadFile("testdata/repaired_text_isolation.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Schema              int                    `json:"schema"`
		SourceRevision      string                 `json:"source_revision"`
		ModelingSHA         string                 `json:"modeling_sha256"`
		TransformersVersion string                 `json:"transformers_version"`
		TransformersSHA     string                 `json:"transformers_qwen_sha256"`
		ConfigSHA           string                 `json:"config_sha256"`
		WeightsSHA          string                 `json:"weights_sha256"`
		WeightsSize         int64                  `json:"weights_size"`
		Policy              string                 `json:"policy"`
		BaseIDs             []int                  `json:"base_ids"`
		StatePositions      []int                  `json:"state_positions"`
		QuestionPositions   [][]int                `json:"question_positions"`
		CandidatePositions  [][][]int              `json:"candidate_positions"`
		ChangedIDs          map[string][]int       `json:"changed_ids"`
		Logits              map[string][][]float64 `json:"logits"`
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Schema != 1 || fixture.SourceRevision != "a74d58cd19ec573e83e8e27f9fecd837b8d830fb" ||
		fixture.ModelingSHA != "a8e93f62d92c6748c5d001fef4f9516d6a74b10158d7265f53bab13f1091d458" ||
		fixture.TransformersVersion != "5.17.0" || fixture.TransformersSHA != "762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b" ||
		fixture.ConfigSHA != "1b3fb0dd8ae5a1e334b31bae1cfb2e8212231bd549884819304f29334313f4c9" ||
		fixture.WeightsSHA != "eae27bf03e0e44501316cafab2406b1505732ddf4b836d19fbb8deb642f55f50" ||
		fixture.WeightsSize != 1710234304 || fixture.Policy != "branch-local-text-absolute-positions-bidirectional-full-causal-linear" {
		t.Fatal("unexpected repair fixture provenance")
	}
	row := repairedTextRow()
	if !reflect.DeepEqual(fixture.BaseIDs, []int{10, 15, 21, 22, 31, 32, 41, 42, 51, 52, 61, 62}) ||
		!reflect.DeepEqual(fixture.StatePositions, []int{0, 1, 2, 3}) ||
		!reflect.DeepEqual(fixture.QuestionPositions, [][]int{{4, 5}, {8, 9}}) ||
		!reflect.DeepEqual(fixture.CandidatePositions, [][][]int{{{6}, {7}}, {{10}, {11}}}) ||
		len(fixture.Logits) != 3 || len(fixture.ChangedIDs) != 2 {
		t.Fatal("unexpected repaired fixture geometry")
	}
	caseIDs := map[string][]int{"base": fixture.BaseIDs}
	for name, ids := range fixture.ChangedIDs {
		caseIDs[name] = ids
	}
	for name, ids := range caseIDs {
		if len(ids) != len(fixture.BaseIDs) || len(fixture.Logits[name]) != 2 {
			t.Fatalf("%s invalid case", name)
		}
		for f := range row.Questions {
			for i, pos := range fixture.QuestionPositions[f] {
				row.Questions[f][i] = ids[pos]
			}
			for n, span := range fixture.CandidatePositions[f] {
				row.Candidates[f][n][0] = ids[span[0]]
			}
		}
		calls := 0
		got, err := ScoreBranchLocalText(row, fixtureHeadForBranchTests(t), func(branch TextBranch) ([]float32, error) {
			calls++
			if branch.StateLen != 4 || branch.QuestionLen != 2 || len(branch.IDs) != 7 || len(branch.AbsolutePositions) != 7 {
				t.Fatalf("invalid branch %+v", branch)
			}
			for i, pos := range branch.AbsolutePositions {
				if branch.IDs[i] != ids[pos] {
					t.Fatalf("branch token mismatch at %d", pos)
				}
			}
			// The independent oracle supplies logits; this injected encoder only
			// checks branch layout and owned inputs. It is not Go encoder parity.
			return make([]float32, 7), nil
		})
		if err != nil || calls != 4 || len(got) != 2 {
			t.Fatalf("%s dispatch: %v, %d calls", name, err, calls)
		}
	}
	for _, c := range []struct {
		self, other  string
		changedField int
	}{{"sibling_candidate", "base", 0}, {"other_question", "base", 1}} {
		self, other := fixture.Logits[c.self], fixture.Logits[c.other]
		for f := range self {
			for n, v := range self[f] {
				if math.IsNaN(v) || math.IsInf(v, 0) {
					t.Fatal("non-finite repaired logit")
				}
				if f != c.changedField && v != other[f][n] {
					t.Fatalf("%s leaked to other question %d/%d", c.self, f, n)
				}
			}
		}
	}
	if fixture.Logits["base"][0][0] != fixture.Logits["sibling_candidate"][0][0] ||
		fixture.Logits["base"][0][1] == fixture.Logits["sibling_candidate"][0][1] ||
		reflect.DeepEqual(fixture.Logits["base"][1], fixture.Logits["other_question"][1]) {
		t.Fatal("repaired fixture lacks own-branch sensitivity/sibling isolation")
	}
}

func fixtureHeadForBranchTests(t *testing.T) *HeadWeights {
	t.Helper()
	h, err := NewHeadWeights(1, 1, []float32{1}, []float32{0}, []float32{1}, []float32{1})
	if err != nil {
		t.Fatal(err)
	}
	return h
}

func TestScoreBranchLocalTextOwnershipAndErrors(t *testing.T) {
	row := repairedTextRow()
	h := fixtureHeadForBranchTests(t)
	calls := 0
	encode := func(b TextBranch) ([]float32, error) {
		calls++
		if !reflect.DeepEqual(b.AbsolutePositions, [][]int{{0, 1, 2, 3, 4, 5, 6}, {0, 1, 2, 3, 4, 5, 7}, {0, 1, 2, 3, 8, 9, 10}, {0, 1, 2, 3, 8, 9, 11}}[calls-1]) {
			t.Fatalf("branch %d positions %v", calls, b.AbsolutePositions)
		}
		for i := range b.IDs {
			b.IDs[i] = -1
			b.AbsolutePositions[i] = -1
		}
		return []float32{1, 1, 1, 1, 1, 1, 1}, nil
	}
	got, err := ScoreBranchLocalText(row, h, encode)
	if err != nil || calls != 4 || len(got) != 2 || len(got[0]) != 2 {
		t.Fatalf("got %v, %v, calls=%d", got, err, calls)
	}
	if !reflect.DeepEqual(row, repairedTextRow()) {
		t.Fatal("caller row mutated")
	}
	bad := []EncodedRow{
		{},
		{State: []int{-1}, Questions: [][]int{{1}}, Candidates: [][][]int{{{2}, {3}}}},
		{State: []int{1}, Questions: [][]int{{1}}, Candidates: [][][]int{{{2}, {-3}}}},
		{State: []int{1}, Questions: [][]int{{}}, Candidates: [][][]int{{{2}, {3}}}},
		{State: []int{1}, Questions: [][]int{{1}}, Candidates: [][][]int{{{2}}}},
		{State: []int{1}, Questions: [][]int{{1}}, Candidates: [][][]int{{{2}, {}}}},
		{State: make([]int, 4095), Questions: [][]int{{1}}, Candidates: [][][]int{{{2}, {3}}}},
	}
	for i, input := range bad {
		called := false
		if out, err := ScoreBranchLocalText(input, h, func(TextBranch) ([]float32, error) { called = true; return nil, nil }); err == nil || out != nil || called {
			t.Fatalf("bad %d got %v, %v called=%v", i, out, err, called)
		}
	}
	for _, tc := range []struct {
		name    string
		backend TextBranchEncoder
	}{
		{"error", func(TextBranch) ([]float32, error) { return nil, errors.New("backend failure") }},
		{"short", func(TextBranch) ([]float32, error) { return []float32{1}, nil }},
		{"nonfinite", func(TextBranch) ([]float32, error) { return []float32{1, 1, 1, 1, 1, 1, float32(math.NaN())}, nil }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if out, err := ScoreBranchLocalText(row, h, tc.backend); err == nil || out != nil {
				t.Fatalf("%v, %v", out, err)
			}
		})
	}
	if out, err := ScoreBranchLocalText(row, h, nil); err == nil || out != nil {
		t.Fatalf("nil backend: %v,%v", out, err)
	}
	if out, err := ScoreBranchLocalText(row, nil, encode); err == nil || out != nil {
		t.Fatalf("nil head: %v,%v", out, err)
	}
}

func TestScoreBranchLocalTextSyntheticIsolation(t *testing.T) {
	row := repairedTextRow()
	h, err := NewHeadWeights(2, 2, []float32{1, 1}, []float32{0, 0}, []float32{1, 0, 0, 1}, []float32{1, 0, 0, 1})
	if err != nil {
		t.Fatal(err)
	}
	encode := func(b TextBranch) ([]float32, error) {
		// This test encoder observes only the branch it was given. It is a
		// synthetic isolation test, not released encoder parity.
		hidden := make([]float32, 2*len(b.IDs))
		for i, id := range b.IDs {
			hidden[2*i] = float32(id)
			hidden[2*i+1] = float32(100 - id)
		}
		return hidden, nil
	}
	base, err := ScoreBranchLocalText(row, h, encode)
	if err != nil {
		t.Fatal(err)
	}
	row.Candidates[0][1][0] = 123
	changed, err := ScoreBranchLocalText(row, h, encode)
	if err != nil {
		t.Fatal(err)
	}
	if base[0][0] != changed[0][0] || !reflect.DeepEqual(base[1], changed[1]) {
		t.Fatalf("sibling changed unrelated logits: %v -> %v", base, changed)
	}
	row = repairedTextRow()
	row.Questions[1][0] = 124
	changed, err = ScoreBranchLocalText(row, h, encode)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(base[0], changed[0]) {
		t.Fatalf("question changed unrelated logits: %v -> %v", base, changed)
	}
}

func TestScoreBranchLocalTextConcurrent(t *testing.T) {
	row := repairedTextRow()
	h := fixtureHeadForBranchTests(t)
	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 16 {
				got, err := ScoreBranchLocalText(row, h, func(b TextBranch) ([]float32, error) { return make([]float32, len(b.IDs)), nil })
				if err != nil || len(got) != 2 {
					t.Errorf("concurrent %v %v", got, err)
				}
			}
		}()
	}
	wg.Wait()
}
