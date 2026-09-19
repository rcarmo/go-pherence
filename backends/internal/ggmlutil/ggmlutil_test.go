package ggmlutil

import "testing"

func TestRawBytesRequiresWholeBlocksAndCheckedProduct(t *testing.T) {
	for _, c := range [][4]int{{32, 32, 34, 34}, {64, 32, 34, 68}, {33, 32, 34, 0}, {-32, 32, 34, 0}, {0, 32, 34, 0}, {32, 0, 34, 0}, {32, 32, -1, 0}, {int(^uint(0) >> 1), 1, 2, 0}} {
		if got := RawBytes(c[0], c[1], c[2]); got != c[3] {
			t.Fatalf("%v got=%d", c, got)
		}
	}
}
