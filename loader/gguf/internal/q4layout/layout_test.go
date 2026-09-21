package q4layout

import (
	"math"
	"testing"
)

func TestTileCheckedProductsAndCounter(t *testing.T) {
	for _, panels := range []int{1, 2, 4} {
		if !Tile(144, 136*panels, 1, panels) {
			t.Fatal(panels)
		}
	}
	maxInt := int(^uint(0) >> 1)
	for _, blocks := range []int{0, -1, maxInt/8 + 1, maxInt} {
		if Tile(0, 0, blocks, 1) {
			t.Fatal("wrapped sizes", blocks)
		}
	}
	if Tile(144, 135, 1, 1) || Tile(144, 136, 1, 0) {
		t.Fatal("invalid tile accepted")
	}
}
func TestRowsRejectWrapAndNarrowing(t *testing.T) {
	if !Rows(288, 272, 45, 0, 2, 9, 5, 1) || !Rows(144, 272, 45, 8, 1, 9, 5, 1) {
		t.Fatal("padded tails rejected")
	}
	for _, dims := range [][5]int{{0, 1, 8, math.MaxInt32, 1}, {0, 1, math.MaxInt32, 4, 1}, {8, 2, 9, 5, 1}, {1, 1, 9, 5, 1}, {0, 1, 8, 4, int(^uint(0) >> 1)}} {
		if Rows(144, 136, 32, dims[0], dims[1], dims[2], dims[3], dims[4]) {
			t.Fatal(dims)
		}
	}
	maxInt := int(^uint(0) >> 1)
	if Groups(maxInt, 8) != maxInt/8+1 {
		t.Fatal("rounded groups overflow")
	}
}
