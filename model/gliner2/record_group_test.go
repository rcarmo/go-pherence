package gliner2

import (
	"reflect"
	"testing"
)

func TestRecordHeadForwardGroupDenseNaturalUsesAnchorMembershipAndSafeQueries(t *testing.T) {
	head := testRecordHeadScalar()
	spec := RecordSpec{
		Mode:          RecordModeNatural,
		AnchorQueryID: 0,
		Fields: []RecordField{
			{QueryID: 0, Name: "anchor", Scalar: true},
			{QueryID: 99, Name: "detail", Scalar: true},
		},
	}
	queries := [][]float32{{10}, {20}}
	candidateStates := [][]float32{{2}, {5}, {7}}
	pool := PooledCandidates{
		Indices:        [][]int{{0, 1}, {1, 2}, {2, 3}},
		ProposalLogits: []float32{1, 2, 3},
		CompatLogits:   []float32{4, 5, 6},
		ValidMask:      []bool{true, false, true},
	}
	pairLogits := [][]float32{{1, 11}, {2, 12}, {3, 13}}
	got, err := head.ForwardGroupDense(spec, queries, candidateStates, pool, pairLogits, []bool{true, true})
	if err != nil {
		t.Fatal(err)
	}
	assertSliceClose(t, got.ObjectLogits, []float32{1, 2, 3}, 1e-6)
	assertBoolSliceEqual(t, got.InstanceMask, []bool{true, false, true})
	assertIntSliceEqual(t, got.InstancePoolIndex, []int{0, 1, 2})
	assertIntSliceEqual(t, got.FieldQueryIDs, []int{0, 99})
	assertBoolMatrixEqual(t, got.FieldMembership, [][]bool{{true, false, true}, {false, false, false}})
	if !reflect.DeepEqual(got.PoolSpans, pool.Indices) {
		t.Fatalf("pool spans=%v want=%v", got.PoolSpans, pool.Indices)
	}
	wantAssign := [][][]float32{
		{{12, 24, MaskLogit, 84}, {22, MaskLogit, MaskLogit, MaskLogit}},
		{{15, 30, MaskLogit, 105}, {25, MaskLogit, MaskLogit, MaskLogit}},
		{{17, 34, MaskLogit, 119}, {27, MaskLogit, MaskLogit, MaskLogit}},
	}
	assertTensor3Close(t, got.AssignLogits, wantAssign, 1e-6)
}

func TestRecordHeadForwardGroupDenseLatentUsesAnyFieldMembership(t *testing.T) {
	head := testRecordHeadScalar()
	spec := RecordSpec{
		Mode: RecordModeLatent,
		Fields: []RecordField{
			{QueryID: 0, Name: "a", Scalar: true},
			{QueryID: 1, Name: "b", Scalar: true},
		},
	}
	queries := [][]float32{{10}, {20}}
	candidateStates := [][]float32{{2}, {5}, {7}}
	pool := PooledCandidates{
		Indices:        [][]int{{0, 1}, {1, 2}, {2, 3}},
		ProposalLogits: []float32{1, 2, 3},
		CompatLogits:   []float32{4, 5, 6},
		ValidMask:      []bool{true, false, true},
	}
	pairLogits := [][]float32{{1, 11}, {2, 12}, {3, 13}}
	got, err := head.ForwardGroupDense(spec, queries, candidateStates, pool, pairLogits, []bool{false, true})
	if err != nil {
		t.Fatal(err)
	}
	assertSliceClose(t, got.ObjectLogits, []float32{2, 5, 7}, 1e-6)
	assertBoolSliceEqual(t, got.InstanceMask, []bool{true, false, true})
	assertIntSliceEqual(t, got.InstancePoolIndex, []int{0, 1, 2})
	assertBoolMatrixEqual(t, got.FieldMembership, [][]bool{{false, false, false}, {true, false, true}})
	wantAssign := [][][]float32{
		{{12, MaskLogit, MaskLogit, MaskLogit}, {22, 44, MaskLogit, 154}},
		{{15, MaskLogit, MaskLogit, MaskLogit}, {25, 50, MaskLogit, 175}},
		{{17, MaskLogit, MaskLogit, MaskLogit}, {27, 54, MaskLogit, 189}},
	}
	assertTensor3Close(t, got.AssignLogits, wantAssign, 1e-6)
}

func TestRecordHeadForwardGroupDenseAnchorlessMaskedAttention(t *testing.T) {
	head := testRecordHeadScalar()
	spec := RecordSpec{
		Mode: RecordModeAnchorless,
		Fields: []RecordField{
			{QueryID: 0, Name: "value", Scalar: true},
			{QueryID: 99, Name: "invalid", Scalar: true},
		},
	}
	queries := [][]float32{{3}, {4}}
	candidateStates := [][]float32{{10}, {20}}
	pool := PooledCandidates{
		Indices:        [][]int{{0, 1}, {1, 2}},
		ProposalLogits: []float32{1, 2},
		CompatLogits:   []float32{3, 4},
		ValidMask:      []bool{true, false},
	}
	pairLogits := [][]float32{{0, 0}, {0, 0}}
	got, err := head.ForwardGroupDense(spec, queries, candidateStates, pool, pairLogits, []bool{true, true})
	if err != nil {
		t.Fatal(err)
	}
	assertSliceClose(t, got.ObjectLogits, []float32{11, 12}, 1e-5)
	assertBoolSliceEqual(t, got.InstanceMask, []bool{true, true})
	assertIntSliceEqual(t, got.InstancePoolIndex, []int{-1, -1})
	assertBoolMatrixEqual(t, got.FieldMembership, [][]bool{{true, false}, {false, false}})
	wantAssign := [][][]float32{
		{{14, 140, MaskLogit}, {15, MaskLogit, MaskLogit}},
		{{15, 150, MaskLogit}, {16, MaskLogit, MaskLogit}},
	}
	assertTensor3Close(t, got.AssignLogits, wantAssign, 1e-5)
}

func TestRecordHeadForwardGroupDenseAnchorlessAllMaskedUsesFiniteMask(t *testing.T) {
	head := testRecordHeadScalar()
	spec := RecordSpec{
		Mode: RecordModeAnchorless,
		Fields: []RecordField{
			{QueryID: 0, Name: "value", Scalar: true},
		},
	}
	queries := [][]float32{{3}}
	candidateStates := [][]float32{{10}, {20}}
	pool := PooledCandidates{
		Indices:        [][]int{{0, 1}, {1, 2}},
		ProposalLogits: []float32{1, 2},
		CompatLogits:   []float32{3, 4},
		ValidMask:      []bool{false, false},
	}
	pairLogits := [][]float32{{0}, {0}}
	got, err := head.ForwardGroupDense(spec, queries, candidateStates, pool, pairLogits, []bool{true})
	if err != nil {
		t.Fatal(err)
	}
	assertSliceClose(t, got.ObjectLogits, []float32{16, 17}, 1e-5)
	assertBoolSliceEqual(t, got.InstanceMask, []bool{true, true})
	assertBoolMatrixEqual(t, got.FieldMembership, [][]bool{{false, false}})
	wantAssign := [][][]float32{
		{{19, MaskLogit, MaskLogit}},
		{{20, MaskLogit, MaskLogit}},
	}
	assertTensor3Close(t, got.AssignLogits, wantAssign, 1e-5)
}

func TestRecordHeadForwardGroupDenseRejectsShapeMismatch(t *testing.T) {
	head := testRecordHeadScalar()
	spec := RecordSpec{Mode: RecordModeLatent, Fields: []RecordField{{QueryID: 0, Name: "a", Scalar: true}}}
	_, err := head.ForwardGroupDense(
		spec,
		[][]float32{{1}},
		[][]float32{{2}, {3}},
		PooledCandidates{Indices: [][]int{{0, 1}}, ValidMask: []bool{true}},
		[][]float32{{1}},
		[]bool{true},
	)
	if err == nil {
		t.Fatal("shape mismatch accepted")
	}
}

func assertBoolSliceEqual(t *testing.T, got, want []bool) {
	t.Helper()
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("bools=%v want=%v", got, want)
	}
}

func assertBoolMatrixEqual(t *testing.T, got, want [][]bool) {
	t.Helper()
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("bool matrix=%v want=%v", got, want)
	}
}

func assertIntSliceEqual(t *testing.T, got, want []int) {
	t.Helper()
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ints=%v want=%v", got, want)
	}
}
