package training

import "fmt"

// GELUShape bounds tanh-approximate GELU (mode0) or its input VJP (mode1).
func GELUShape(n int, mode uint32) error {
	if n < 1 || n > 1<<26 || mode > 1 {
		return fmt.Errorf("invalid training GELU size/mode")
	}
	return nil
}

// RowShape describes disjoint overwrite operations for transformer rows.
// mode0 gathers one block from [rows,parts,width] to [rows,width]; mode1
// scatters [rows,width] to that block and zeroes the rest. mode2 multiplies
// [rows,width] by a shared [width] scale; mode3 reduces X*dY over rows into
// [width]. Modes0/1 bind an unused owned B[width]; parts=1/slot=0 for modes2/3.
// Returned extents are A, B and output elements. No in-place writes/atomics.
func RowShape(rows, width, parts, slot int, mode uint32) (a, b, out int, err error) {
	if rows < 1 || rows > 1024 || width < 1 || width > 16384 || parts < 1 || parts > 4 || slot < 0 || slot >= parts || mode > 3 || ((mode >= 2) && (parts != 1 || slot != 0)) {
		return 0, 0, 0, fmt.Errorf("invalid training row operation")
	}
	a, b, out = rows*width, width, rows*width
	switch mode {
	case 0:
		a *= parts
	case 1:
		out *= parts
	case 3:
		b = rows * width
		out = width
	}
	return a, b, out, nil
}
