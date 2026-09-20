// Package q4layout validates extents before the experimental C/Plan9 kernels.
package q4layout

import (
	"math"

	"github.com/rcarmo/go-pherence/internal/checked"
)

func product(values ...int) (int, bool) {
	n := 1
	for _, v := range values {
		var ok bool
		n, ok = checked.MulInt(n, v)
		if !ok {
			return 0, false
		}
	}
	return n, true
}

// Groups rounds up without adding to an untrusted dimension.
func Groups(n, lanes int) int {
	if n <= 0 || lanes <= 0 {
		return 0
	}
	return 1 + (n-1)/lanes
}

// Tile bounds the signed C/translated-kernel loop counter and both input spans.
func Tile(q4Bytes, q8Bytes, blocks, panels int) bool {
	if blocks <= 0 || blocks > math.MaxInt32 || panels <= 0 || panels > 4 {
		return false
	}
	w, okW := product(blocks, 144)
	x, okX := product(blocks, 136, panels)
	return okW && okX && q4Bytes == w && q8Bytes == x
}

// Rows leaves room for C's rounded rows/tokens and loop increments. Bounds
// apply equally to Plan9's retained compiler-derived ABI. Lengths are exact;
// tail lanes must be present in packed inputs but never written to output.
func Rows(q4Bytes, q8Bytes, outFloats, rowBase, rowGroups, rows, tokens, blocks int) bool {
	if rows <= 0 || rows > math.MaxInt32-7 || tokens <= 0 || tokens > math.MaxInt32-15 || blocks <= 0 || blocks > math.MaxInt32 || rowBase < 0 || rowBase%8 != 0 || rowBase >= rows || rowGroups <= 0 {
		return false
	}
	groups := Groups(rows, 8)
	if rowGroups > groups-rowBase/8 {
		return false
	}
	w, okW := product(rowGroups, blocks, 144)
	x, okX := product(Groups(tokens, 4), blocks, 136)
	out, okOut := product(rows, tokens)
	_, okBytes := product(out, 4)
	return okW && okX && okOut && okBytes && q4Bytes == w && q8Bytes == x && outFloats == out
}
