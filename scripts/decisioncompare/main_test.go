package main

import (
	"encoding/json"
	"math"
	"testing"

	gso "github.com/rcarmo/go-pherence/model/gosystemone"
)

func example(logits []float32, p float64) trial {
	winner := json.RawMessage(`true`)
	if p < .5 {
		winner = json.RawMessage(`false`)
	}
	return trial{Logits: [][][]float32{{logits}}, Response: gso.Response{Results: []gso.Result{{Fields: map[string]gso.FieldResult{"urgent": {Value: winner, Candidates: []gso.CandidateResult{{Value: json.RawMessage(`true`), Probability: p}, {Value: json.RawMessage(`false`), Probability: 1 - p}}}}}}}}
}
func TestCompareCommonShift(t *testing.T) {
	a, b := example([]float32{1, 2}, .25), example([]float32{11, 12}, .25)
	c, err := compare(a, b, []float64{.5, .9})
	if err != nil {
		t.Fatal(err)
	}
	if c.Raw.Max != 10 || c.Raw.RMS != 10 || c.Centred.Max != 0 || c.ChangedFields != 0 || c.Fields[0].Probability.Max != 0 {
		t.Fatalf("%+v", c)
	}
}
func TestCompareNearTieCrossing(t *testing.T) {
	c, err := compare(example([]float32{1, 1.01}, .49), example([]float32{1.01, 1}, .51), []float64{.5, .9})
	if err != nil {
		t.Fatal(err)
	}
	if c.ChangedFields != 1 || len(c.Fields[0].ThresholdCrossings) != 2 || math.Abs(c.Fields[0].SerialMargin-.02) > 1e-12 || math.Abs(c.Fields[0].Probability.Max-.02) > 1e-12 {
		t.Fatalf("%+v", c)
	}
}
func TestCompareMalformed(t *testing.T) {
	a := example([]float32{1, 2}, .25)
	for _, b := range []trial{{}, example([]float32{1}, .25), example([]float32{1, float32(math.NaN())}, .25)} {
		if _, err := compare(a, b, nil); err == nil {
			t.Fatal("malformed accepted")
		}
	}
}

func TestCompareLowerRanksChangeWithoutWinner(t *testing.T) {
	a, b := example([]float32{3, 2, 1}, .8), example([]float32{3, 1, 2}, .8)
	af := a.Response.Results[0].Fields["urgent"]
	bf := b.Response.Results[0].Fields["urgent"]
	af.Candidates = []gso.CandidateResult{{Value: json.RawMessage(`"a"`), Probability: .8}, {Value: json.RawMessage(`"b"`), Probability: .15}, {Value: json.RawMessage(`"c"`), Probability: .05}}
	bf.Candidates = []gso.CandidateResult{{Value: json.RawMessage(`"a"`), Probability: .8}, {Value: json.RawMessage(`"b"`), Probability: .05}, {Value: json.RawMessage(`"c"`), Probability: .15}}
	af.Value = json.RawMessage(`"a"`)
	bf.Value = af.Value
	a.Response.Results[0].Fields["urgent"] = af
	b.Response.Results[0].Fields["urgent"] = bf
	c, err := compare(a, b, nil)
	if err != nil {
		t.Fatal(err)
	}
	if c.ChangedFields != 0 || !c.Fields[0].RankChanged || c.Fields[0].PackedRank[1] != 2 {
		t.Fatalf("%+v", c)
	}
}

func TestResumeChecksRequestsAndProvenance(t *testing.T) {
	r := gso.Request{Schema: json.RawMessage(`{"x":{"type":"boolean","description":"x"}}`), Contexts: []string{"a", "b"}, Mode: gso.ModeTree}
	if err := r.NormalizeAndValidate(); err != nil {
		t.Fatal(err)
	}
	tr := example([]float32{1, 2}, .25)
	cr := caseReport{Request: r, Trials: []trial{tr, tr, tr, tr}, Comparisons: make([]comparison, 3)}
	for i, n := range []int{0, 128, 256, 512} {
		cr.Trials[i].Rows = n
	}
	old := report{Source: "sha", CohortSHA: "cohort", ModelSHA: "model", Cases: []caseReport{cr}}
	path := t.TempDir() + "/report.json"
	if err := save(path, old); err != nil {
		t.Fatal(err)
	}
	c := cohort{Requests: []gso.Request{r}}
	target := old
	target.Cases = nil
	if err := resumeReport(path, &target, c, 2); err != nil {
		t.Fatal(err)
	}
	target.Source = "different"
	if err := resumeReport(path, &target, c, 2); err == nil {
		t.Fatal("changed source admitted")
	}
	target = old
	if err := resumeReport(path, &target, c, 1); err == nil {
		t.Fatal("changed chunk admitted")
	}
	c.Requests[0].Contexts = []string{"different", "b"}
	if err := resumeReport(path, &target, c, 2); err == nil {
		t.Fatal("changed request admitted")
	}
}
