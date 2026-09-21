package gliner2

import "testing"

func TestExclusiveScalarGlobalAssignment(t *testing.T) {
	g := DenseRecordGroupOutput{Fields: []RecordField{{Scalar: true, Required: true, Exclusive: true}}, FieldMembership: [][]bool{{true, true}}, AssignLogits: [][][]float32{{{-10, 5, 4}}, {{-10, 6, 0}}}}
	got, err := exclusiveRecordSelections(g, []int{0, 1}, 1, .5)
	if err != nil {
		t.Fatal(err)
	}
	if len(got[[2]int{0, 0}]) != 1 || got[[2]int{0, 0}][1] == 0 || got[[2]int{1, 0}][0] == 0 {
		t.Fatal("greedy assignment instead of global", got)
	}
	g.Fields[0].Required = false
	g.AssignLogits[0][0][0] = 20
	g.AssignLogits[1][0][0] = 20
	got, err = exclusiveRecordSelections(g, []int{0, 1}, 1, .5)
	if err != nil || len(got) != 0 {
		t.Fatal("absent assignment", got, err)
	}
}
func TestExclusiveListSingleOwner(t *testing.T) {
	g := DenseRecordGroupOutput{Fields: []RecordField{{Exclusive: true}}, FieldMembership: [][]bool{{true, false}}, AssignLogits: [][][]float32{{{0, 2, 9}}, {{0, 3, 10}}}}
	got, err := exclusiveRecordSelections(g, []int{0, 1}, 1, .5)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[[2]int{1, 0}][0] == 0 {
		t.Fatal(got)
	}
}
