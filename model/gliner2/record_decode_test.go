package gliner2

import "testing"

func TestRecordDecodeNullAndAnchor(t *testing.T) {
	text := "Ada London"
	words, _ := SplitWords(text)
	fields := []RecordField{{QueryID: 0, Name: "name", Scalar: true}, {QueryID: 1, Name: "city", Scalar: true}}
	spec := RecordSpec{Mode: RecordModeNatural, AnchorQueryID: 0, Fields: fields}
	s := RecordScores{Input: EntityInput{Words: words}, CandidateLogits: [][]float32{{5, -5}, {-5, 5}}, Group: DenseRecordGroupOutput{Spec: spec, Fields: fields, ObjectLogits: []float32{5}, InstanceMask: []bool{true}, InstancePoolIndex: []int{0}, PoolSpans: [][]int{{0, 1}, {1, 2}}, FieldMembership: [][]bool{{true, true}, {true, true}}, AssignLogits: [][][]float32{{{10, -5, -5}, {-5, -5, 5}}}}}
	c := BoundaryHeadConfig{RecordTemperature: 1, RecordAnchorThreshold: .5, RecordFieldThreshold: .5, OverlapPolicy: "flat"}
	got, err := DecodeRecords(text, s, c)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].Fields["name"][0].Text != "Ada" || got[0].Fields["city"][0].Text != "London" {
		t.Fatal(got)
	}
	s.Group.AssignLogits[0][1][0] = 20
	got, err = DecodeRecords(text, s, c)
	if err != nil {
		t.Fatal(err)
	}
	if len(got[0].Fields["city"]) != 0 {
		t.Fatal("null not selected")
	}
	s.Group.AssignLogits[0][1] = nil
	if _, err = DecodeRecords(text, s, c); err == nil {
		t.Fatal("invalid assignment accepted")
	}
}
