package needle

import (
	"math"
	"testing"
)

// The tiny archive's packed embedding row must reproduce the decoded table
// without changing the existing trunk gather or loader representation.
func TestPackedEmbeddingRowMatchesDecodedArchive(t *testing.T) {
	m, _, err := LoadArchive("../../loader/needle/testdata/needle3.cact")
	if err != nil {
		t.Fatal(err)
	}
	packed := m.packed[packedKey{"embedding/embedding", -1}]
	dense := m.tensors["embedding/embedding"]
	if packed == nil || len(dense.Shape) != 2 || packed.Rows() != dense.Shape[0] || packed.Cols() != dense.Shape[1] {
		t.Fatal("missing packed/decoded embedding geometry")
	}
	for row := 0; row < packed.Rows(); row++ {
		got, err := packed.DecodeRow(row)
		if err != nil {
			t.Fatal(err)
		}
		for col, v := range got {
			want := dense.Data[row*packed.Cols()+col]
			if math.Float32bits(v) != math.Float32bits(want) {
				t.Fatalf("embedding row=%d col=%d got=%g want=%g", row, col, v, want)
			}
		}
	}
	for _, row := range []int{-1, packed.Rows()} {
		if got, err := packed.DecodeRow(row); err == nil || got != nil {
			t.Fatalf("accepted row %d", row)
		}
	}
}
