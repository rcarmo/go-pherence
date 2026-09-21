package gliner2

import (
	"encoding/json"
	"math"
	"os"
	"strconv"
	"testing"
)

// Uses retained published-model logits so default CI checks final field
// selection without requiring the 774MB checkpoint or a Python installation.
func TestPublishedRecordDecodeParity(t *testing.T) {
	var scores struct {
		Text   string
		IDs    []int
		Spans  [][]int
		Mask   []bool
		Object []float32
		Assign map[string][][]float32
	}
	var oracle struct {
		Text            string
		CandidateLogits [][]float32 `json:"candidate_logits"`
		Decoded         []struct {
			Score       float64
			Fields      map[string][][]int
			FieldScores map[string][]float64 `json:"field_scores"`
		}
	}
	load := func(path string, v any) {
		t.Helper()
		raw, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		if err = json.Unmarshal(raw, v); err != nil {
			t.Fatal(err)
		}
	}
	load("testdata/published_record_reference.json", &scores)
	load("testdata/published_record_decode_reference.json", &oracle)
	words, err := SplitWords(scores.Text)
	if err != nil {
		t.Fatal(err)
	}
	fields := []RecordField{{QueryID: 0, Name: "name", Scalar: true}, {QueryID: 1, Name: "city", Scalar: true}}
	group := DenseRecordGroupOutput{Spec: RecordSpec{Mode: RecordModeNatural, AnchorQueryID: 0, Fields: fields}, Fields: fields, ObjectLogits: scores.Object, InstanceMask: scores.Mask, PoolSpans: scores.Spans, InstancePoolIndex: make([]int, len(scores.Mask)), AssignLogits: make([][][]float32, len(scores.Mask)), FieldMembership: [][]bool{scores.Mask, scores.Mask}}
	for i := range scores.Mask {
		group.InstancePoolIndex[i] = i
		group.AssignLogits[i] = scores.Assign[strconv.Itoa(i)]
		if group.AssignLogits[i] == nil {
			group.AssignLogits[i] = [][]float32{make([]float32, len(scores.Spans)+1), make([]float32, len(scores.Spans)+1)}
		}
	}
	c := BoundaryHeadConfig{RecordTemperature: 1, RecordAnchorThreshold: .5, RecordFieldThreshold: .5, OverlapPolicy: "flat"}
	got, err := DecodeRecords(scores.Text, RecordScores{Input: EntityInput{Words: words}, Group: group, CandidateLogits: oracle.CandidateLogits}, c)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(oracle.Decoded) {
		t.Fatalf("records %d want %d", len(got), len(oracle.Decoded))
	}
	for i, r := range got {
		expected := oracle.Decoded[i]
		if math.Abs(r.Confidence-expected.Score) > 2e-6 {
			t.Fatal("record score mismatch")
		}
		for q, field := range fields {
			spans := expected.Fields[strconv.Itoa(q)]
			actual := r.Fields[field.Name]
			if len(actual) != len(spans) {
				t.Fatalf("field %s count mismatch", field.Name)
			}
			for j, e := range actual {
				span := spans[j]
				if e.TokenStart != span[0] || e.TokenEnd != span[1] {
					t.Fatalf("field %s span mismatch", field.Name)
				}
				// engine formatting caps assignment confidence by candidate probability.
				candidate := float64(0)
				for k, p := range scores.Spans {
					if p[0] == span[0] && p[1] == span[1] {
						candidate = sigmoid(float64(oracle.CandidateLogits[k][q]))
						break
					}
				}
				want := math.Min(candidate, expected.FieldScores[strconv.Itoa(q)][j])
				if math.Abs(e.Confidence-want) > 2e-6 {
					t.Fatalf("field %s score %g want %g", field.Name, e.Confidence, want)
				}
			}
		}
	}
}
