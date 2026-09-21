package trellis2

import "testing"

func TestSparseTensorRejectsWrappedStorage(t *testing.T) {
	rows := int(^uint(0)>>1)/2 + 1
	if _, err := NewSparseTensor(nil, nil, rows, 4); err == nil {
		t.Fatal("wrapped counts accepted")
	}
	s := SparseTensor{Rows: 2, Dim: int(^uint(0) >> 1), Coords: make([]int32, 8)}
	if s.Validate() == nil {
		t.Fatal("wrapped features accepted")
	}
}
func TestSparseAccessorsRejectForgedViews(t *testing.T) {
	for _, s := range []SparseTensor{{Rows: 1, Dim: 1}, {Rows: 2, Dim: 4, Coords: make([]int32, 4), Feats: make([]float32, 4)}, {Rows: 1, Dim: int(^uint(0) >> 1)}} {
		if _, err := s.Coord(s.Rows - 1); err == nil {
			t.Fatal("short coords accepted")
		}
		if _, err := s.FeatureRow(s.Rows - 1); err == nil {
			t.Fatal("short features accepted")
		}
	}
}
func TestSparseLinearRejectsWrappedWeights(t *testing.T) {
	s, err := NewSparseTensor(make([]int32, 4), []float32{1, 2, 3, 4}, 1, 4)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := SparseLinearFloat32(s, nil, nil, int(^uint(0)>>1)/2+1); err == nil {
		t.Fatal("wrapped projection accepted")
	}
}
