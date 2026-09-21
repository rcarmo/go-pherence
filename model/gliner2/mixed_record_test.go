package gliner2

import (
	"reflect"
	"testing"
)

func TestGlobalRecordSpecForSchemaMapsLocalQueryIDsAfterEntityGroup(t *testing.T) {
	schema := TextSchema{
		Parent: "person",
		Marker: "[C]",
		Labels: []string{"name", "city"},
		Record: &RecordSpec{
			Mode:          RecordModeNatural,
			AnchorQueryID: 0,
			Fields:        []RecordField{{QueryID: 0, Name: "name", Scalar: true}, {QueryID: 1, Name: "city", Scalar: true}},
		},
	}
	got, err := globalRecordSpecForSchema(schema, []int{1, 2})
	if err != nil {
		t.Fatal(err)
	}
	want := RecordSpec{
		Mode:          RecordModeNatural,
		AnchorQueryID: 1,
		Fields:        []RecordField{{QueryID: 1, Name: "name", Scalar: true}, {QueryID: 2, Name: "city", Scalar: true}},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("mapped spec=%+v want %+v", got, want)
	}
	head := testRecordHeadScalar()
	group, err := head.ForwardGroupDense(
		got,
		[][]float32{{9}, {1}, {2}},
		[][]float32{{5}, {6}},
		PooledCandidates{Indices: [][]int{{0, 1}, {1, 2}}, ValidMask: []bool{true, true}},
		[][]float32{{111, 11, 21}, {222, 12, 22}},
		[]bool{true, true, true},
	)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(group.FieldQueryIDs, []int{1, 2}) {
		t.Fatal(group.FieldQueryIDs)
	}
	if !reflect.DeepEqual(group.ObjectLogits, []float32{11, 12}) {
		t.Fatalf("anchor logits=%v want [11 12]", group.ObjectLogits)
	}
}

func TestValidateTextSchemaRecordMetadataRejectsMismatchAndNonC(t *testing.T) {
	cases := []TextSchema{
		{Parent: "entities", Marker: "[E]", Labels: []string{"name"}, Record: &RecordSpec{Mode: RecordModeNatural, AnchorQueryID: 0, Fields: []RecordField{{QueryID: 0, Name: "name", Scalar: true}}}},
		{Parent: "person", Marker: "[C]", Labels: []string{"name", "city"}, Record: &RecordSpec{Mode: RecordModeNatural, AnchorQueryID: 0, Fields: []RecordField{{QueryID: 0, Name: "name", Scalar: true}, {QueryID: 1, Name: "place", Scalar: true}}}},
		{Parent: "person", Marker: "[C]", Labels: []string{"name", "city"}, Record: &RecordSpec{Mode: RecordModeNatural, AnchorQueryID: 0, Fields: []RecordField{{QueryID: 0, Name: "name", Scalar: true}}}},
	}
	for _, tc := range cases {
		if err := validateTextSchemaRecordMetadata(tc); err == nil {
			t.Fatalf("accepted %+v", tc)
		}
	}
}

func TestDecodeSchemaGroupsDispatchesRecords(t *testing.T) {
	text := "Ada London"
	decoded, err := DecodeSchemaGroups(text, syntheticMixedRecordScores(t), .5, "flat", BoundaryHeadConfig{PairTemperature: 1, RecordTemperature: 1, RecordAnchorThreshold: .5, RecordFieldThreshold: .5, OverlapPolicy: "flat"})
	if err != nil {
		t.Fatal(err)
	}
	if decoded[0].Entities[0].Text != "Ada" {
		t.Fatal(decoded[0].Entities)
	}
	if len(decoded[1].Entities) != 0 || len(decoded[1].Records) != 1 {
		t.Fatal(decoded[1])
	}
	rec := decoded[1].Records[0]
	if rec.Fields["name"][0].Text != "Ada" || rec.Fields["city"][0].Text != "London" {
		t.Fatal(rec)
	}
}

func TestDecodeSchemaGroupsRejectsRawAndMismatchedRecordGroups(t *testing.T) {
	scores := syntheticMixedRecordScores(t)
	scores.Input.Groups[1].Schema.Record = nil
	if _, err := DecodeSchemaGroups("Ada London", scores, .5, "flat", BoundaryHeadConfig{PairTemperature: 1, RecordTemperature: 1, RecordAnchorThreshold: .5, RecordFieldThreshold: .5, OverlapPolicy: "flat"}); err == nil {
		t.Fatal("raw [C] group decoded without metadata")
	}
	scores = syntheticMixedRecordScores(t)
	scores.Records[1] = mutatedRecordScores(scores.Records[1], func(r *RecordScores) {
		r.Group.Spec.AnchorQueryID = 0
	})
	if _, err := DecodeSchemaGroups("Ada London", scores, .5, "flat", BoundaryHeadConfig{PairTemperature: 1, RecordTemperature: 1, RecordAnchorThreshold: .5, RecordFieldThreshold: .5, OverlapPolicy: "flat"}); err == nil {
		t.Fatal("mismatched record schema accepted")
	}
}

func syntheticMixedRecordScores(t *testing.T) MixedScores {
	t.Helper()
	text := "Ada London"
	words, err := SplitWords(text)
	if err != nil {
		t.Fatal(err)
	}
	entitySchema := TextSchema{Parent: "entities", Marker: "[E]", Labels: []string{"person"}}
	recordSchema := TextSchema{
		Parent: "person",
		Marker: "[C]",
		Labels: []string{"name", "city"},
		Record: &RecordSpec{
			Mode:          RecordModeNatural,
			AnchorQueryID: 0,
			Fields:        []RecordField{{QueryID: 0, Name: "name", Scalar: true}, {QueryID: 1, Name: "city", Scalar: true}},
		},
	}
	spec, err := globalRecordSpecForSchema(recordSchema, []int{1, 2})
	if err != nil {
		t.Fatal(err)
	}
	logits := [][]float32{{5, 5, -5}, {-5, -5, 5}}
	return MixedScores{
		Input: MixedInput{
			Words:  words,
			Groups: []SchemaGroupRouting{{Schema: entitySchema}, {Schema: recordSchema}},
		},
		GroupQueryIDs: [][]int{{0}, {1, 2}},
		Extraction: &EntityScores{
			Input:      EntityInput{Words: words, Labels: []string{"person", "name", "city"}},
			Candidates: PooledCandidates{Indices: [][]int{{0, 1}, {1, 2}}, ValidMask: []bool{true, true}},
			Logits:     logits,
		},
		Records: map[int]RecordScores{1: {
			Input:           EntityInput{Words: words},
			CandidateLogits: logits,
			Group: DenseRecordGroupOutput{
				Spec:              spec,
				Fields:            append([]RecordField(nil), spec.Fields...),
				FieldQueryIDs:     []int{1, 2},
				ObjectLogits:      []float32{5},
				AssignLogits:      [][][]float32{{{0, 0, 0}, {-5, -5, 5}}},
				InstanceMask:      []bool{true},
				FieldMembership:   [][]bool{{true, true}, {true, true}},
				PoolSpans:         [][]int{{0, 1}, {1, 2}},
				InstancePoolIndex: []int{0},
			},
		}},
	}
}

func mutatedRecordScores(in RecordScores, mutate func(*RecordScores)) RecordScores {
	out := RecordScores{Input: in.Input, CandidateLogits: append([][]float32(nil), in.CandidateLogits...)}
	out.Group = in.Group
	out.Group.Spec = cloneRecordSpec(in.Group.Spec)
	out.Group.Fields = append([]RecordField(nil), in.Group.Fields...)
	out.Group.FieldQueryIDs = append([]int(nil), in.Group.FieldQueryIDs...)
	out.Group.ObjectLogits = append([]float32(nil), in.Group.ObjectLogits...)
	out.Group.InstanceMask = append([]bool(nil), in.Group.InstanceMask...)
	out.Group.InstancePoolIndex = append([]int(nil), in.Group.InstancePoolIndex...)
	out.Group.FieldMembership = append([][]bool(nil), in.Group.FieldMembership...)
	out.Group.PoolSpans = clonePoolSpans(in.Group.PoolSpans)
	out.Group.AssignLogits = make([][][]float32, len(in.Group.AssignLogits))
	for i := range in.Group.AssignLogits {
		out.Group.AssignLogits[i] = make([][]float32, len(in.Group.AssignLogits[i]))
		for j := range in.Group.AssignLogits[i] {
			out.Group.AssignLogits[i][j] = append([]float32(nil), in.Group.AssignLogits[i][j]...)
		}
	}
	for i := range in.Group.FieldMembership {
		out.Group.FieldMembership[i] = append([]bool(nil), in.Group.FieldMembership[i]...)
	}
	mutate(&out)
	return out
}
