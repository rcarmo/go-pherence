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

func TestRequiredScalarSkipsNullAndThreshold(t *testing.T) {
	text := "Ada London"
	words, _ := SplitWords(text)
	fields := []RecordField{{QueryID: 0, Name: "city", Scalar: true, Required: true}}
	spec := RecordSpec{Mode: RecordModeAnchorless, AnchorQueryID: -1, Fields: fields}
	s := RecordScores{Input: EntityInput{Words: words}, CandidateLogits: [][]float32{{-10}, {-10}}, Group: DenseRecordGroupOutput{Spec: spec, Fields: fields, ObjectLogits: []float32{5}, InstanceMask: []bool{true}, InstancePoolIndex: []int{-1}, PoolSpans: [][]int{{0, 1}, {1, 2}}, FieldMembership: [][]bool{{true, true}}, AssignLogits: [][][]float32{{{1000, -5, 1}}}}}
	cfg := BoundaryHeadConfig{RecordTemperature: 1, RecordAnchorThreshold: .5, RecordFieldThreshold: .9, OverlapPolicy: "flat"}
	got, err := DecodeRecords(text, s, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].Fields["city"][0].Text != "London" {
		t.Fatal("required choice dropped", got)
	}
	s.Group.Fields[0].Required = false
	s.Group.Spec.Fields = s.Group.Fields
	got, err = DecodeRecords(text, s, cfg)
	if err != nil || len(got) != 0 {
		t.Fatal("optional null ignored", got, err)
	}
}

func TestRecordDecoderValidatesUnselectedCandidates(t *testing.T) {
	words, _ := SplitWords("Ada")
	field := RecordField{QueryID: 0, Name: "name", Scalar: true}
	makeScores := func() RecordScores {
		return RecordScores{Input: EntityInput{Words: words}, CandidateLogits: [][]float32{{0}}, Group: DenseRecordGroupOutput{Spec: RecordSpec{Mode: RecordModeNatural, AnchorQueryID: 0, Fields: []RecordField{field}}, Fields: []RecordField{field}, PoolSpans: [][]int{{0, 1}}, FieldMembership: [][]bool{{true}}, ObjectLogits: []float32{-100}, InstanceMask: []bool{true}, InstancePoolIndex: []int{0}, AssignLogits: [][][]float32{{{0, 0}}}}}
	}
	cfg := BoundaryHeadConfig{RecordTemperature: 1, RecordAnchorThreshold: .5, RecordFieldThreshold: .5, OverlapPolicy: "flat"}
	s := makeScores()
	s.CandidateLogits[0] = nil
	if _, err := DecodeRecords("Ada", s, cfg); err == nil {
		t.Fatal("unselected bad query shape accepted")
	}
	s = makeScores()
	s.Group.PoolSpans[0] = []int{0}
	if _, err := DecodeRecords("Ada", s, cfg); err == nil {
		t.Fatal("malformed span accepted")
	}
	s = makeScores()
	s.Group.Fields[0].Required = true
	if _, err := DecodeRecords("Ada", s, cfg); err == nil {
		t.Fatal("mismatched schema accepted")
	}
}

func TestRequiredListUsesAssignmentThresholdWithoutCandidateGate(t *testing.T) {
	words, _ := SplitWords("Ada London")
	field := RecordField{QueryID: 0, Name: "places", Required: true}
	fields := []RecordField{field}
	s := RecordScores{Input: EntityInput{Words: words}, CandidateLogits: [][]float32{{-10}, {-10}}, Group: DenseRecordGroupOutput{Spec: RecordSpec{Mode: RecordModeAnchorless, Fields: fields}, Fields: fields, ObjectLogits: []float32{5}, InstanceMask: []bool{true}, InstancePoolIndex: []int{-1}, PoolSpans: [][]int{{0, 1}, {1, 2}}, FieldMembership: [][]bool{{true, true}}, AssignLogits: [][][]float32{{{20, -5, 5}}}}}
	cfg := BoundaryHeadConfig{RecordTemperature: 1, RecordAnchorThreshold: .5, RecordFieldThreshold: .5, OverlapPolicy: "flat"}
	got, err := DecodeRecords("Ada London", s, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || len(got[0].Fields["places"]) != 1 || got[0].Fields["places"][0].Text != "London" {
		t.Fatal(got)
	}
	s.Group.AssignLogits[0][0][2] = -5
	got, err = DecodeRecords("Ada London", s, cfg)
	if err != nil || len(got) != 0 {
		t.Fatal("required list invented fallback", got, err)
	}
}
