package gliner2

import (
	"math"
	"testing"
)

func TestBoundaryQueryHeadMasksAndPrefix(t *testing.T) {
	p := Linear{InDim: 2, OutDim: 2, Weight: []float32{1, 0, 0, 1}}
	h := BoundaryQueryHead{p, p, p, p, p, p}
	r, err := h.Forward([][]float32{{1, 2}, {3, 4}, {5, 6}, {7, 8}}, [][]float32{{2, 0}, {4, 0}, {99, 99}}, [][]float32{{1, 0}, {0, 1}}, []bool{true, true, true, false}, []bool{true, true, false}, []bool{true, false})
	if err != nil {
		t.Fatal(err)
	}
	s := float32(1 / math.Sqrt(2))
	if r.StartLogits[0][0] != s || r.EndLogits[0][3] != MaskLogit || r.InsideLogits[0][2] != MaskLogit {
		t.Fatalf("bad masked scores %+v", r)
	}
	restored, err := IntervalPrefixScore(r.InsidePrefix[0], 0, 2, &r.InsidePrefixMean[0])
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(float64(restored-6*s)) > 1e-6 {
		t.Fatal("prefix mean not restored", restored)
	}
	for _, v := range r.StartLogits[1] {
		if v != MaskLogit {
			t.Fatal("masked query scored")
		}
	}
	for _, v := range r.InsidePrefix[1] {
		if v != 0 {
			t.Fatal("masked query leaked into prefix")
		}
	}
	if r.InsidePrefix[0][3] != r.InsidePrefix[0][2] {
		t.Fatal("masked token leaked")
	}
}
func TestBoundaryQueryHeadValidation(t *testing.T) {
	if (BoundaryQueryHead{}).Validate() == nil {
		t.Fatal("zero head accepted")
	}
	p := Linear{InDim: 2, OutDim: 2, Weight: []float32{1, 0, 0, 1}}
	h := BoundaryQueryHead{p, p, p, p, p, p}
	if _, err := h.Forward(nil, nil, nil, nil, nil, nil); err == nil {
		t.Fatal("bad boundary count accepted")
	}
}
