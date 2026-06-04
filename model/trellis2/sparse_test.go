package trellis2

import "testing"

func TestSparseTensorValidateAndAccessors(t *testing.T) {
	st, err := NewSparseTensor(
		[]int32{0, 1, 2, 3, 0, 4, 5, 6},
		[]float32{1, 2, 3, 4, 5, 6},
		2, 3,
	)
	if err != nil {
		t.Fatal(err)
	}
	coord, err := st.Coord(1)
	if err != nil {
		t.Fatal(err)
	}
	if coord != [4]int32{0, 4, 5, 6} {
		t.Fatalf("coord=%v", coord)
	}
	row, err := st.FeatureRow(1)
	if err != nil {
		t.Fatal(err)
	}
	if len(row) != 3 || row[0] != 4 || row[2] != 6 {
		t.Fatalf("row=%v", row)
	}
	if _, err := NewSparseTensor([]int32{0, -1, 0, 0}, []float32{1}, 1, 1); err == nil {
		t.Fatal("negative coord accepted")
	}
	if _, err := NewSparseTensor([]int32{0, 1, 2}, []float32{1}, 1, 1); err == nil {
		t.Fatal("short coords accepted")
	}
}

func TestSparseLinearFloat32(t *testing.T) {
	src, err := NewSparseTensor(
		[]int32{0, 1, 2, 3, 0, 4, 5, 6},
		[]float32{1, 2, 3, 4, 5, 6},
		2, 3,
	)
	if err != nil {
		t.Fatal(err)
	}
	// weight [out=2,in=3]
	weight := []float32{
		1, 0, 1,
		0, 2, 0,
	}
	bias := []float32{10, -1}
	got, err := SparseLinearFloat32(src, weight, bias, 2)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{14, 3, 20, 9}
	if got.Rows != 2 || got.Dim != 2 {
		t.Fatalf("shape rows=%d dim=%d", got.Rows, got.Dim)
	}
	for i := range want {
		if got.Feats[i] != want[i] {
			t.Fatalf("feats[%d]=%v want %v full=%v", i, got.Feats[i], want[i], got.Feats)
		}
	}
	if got.Coords[4] != 0 || got.Coords[7] != 6 {
		t.Fatalf("coords not preserved: %v", got.Coords)
	}
	got.Coords[0] = 99
	if src.Coords[0] == 99 {
		t.Fatal("output coords alias input coords")
	}
}

func TestSparseLinearFloat32RejectsBadBuffers(t *testing.T) {
	src, err := NewSparseTensor([]int32{0, 1, 2, 3}, []float32{1, 2}, 1, 2)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := SparseLinearFloat32(src, []float32{1}, nil, 1); err == nil {
		t.Fatal("short weight accepted")
	}
	if _, err := SparseLinearFloat32(src, []float32{1, 2}, []float32{}, 1); err == nil {
		t.Fatal("short bias accepted")
	}
	if _, err := SparseLinearFloat32(src, []float32{1, 2}, nil, 0); err == nil {
		t.Fatal("bad outDim accepted")
	}
}
