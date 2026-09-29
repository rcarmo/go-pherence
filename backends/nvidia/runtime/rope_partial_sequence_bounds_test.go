package nvidia

import (
	"math"
	"testing"
)

func TestRoPEPartialSequenceRejectMalformed(t *testing.T) {
	const rows, heads, dim, half = 138, 8, 64, 32
	x := &Buffer{Ptr: 0x10000, Size: rows * heads * dim * 4}
	freq := &Buffer{Ptr: 0x90000, Size: rows * half * 2 * 4}
	maxInt := int(^uint(0) >> 1)
	for _, tc := range []struct {
		name                         string
		x, freq                      *Buffer
		rows, pos0, heads, dim, half int
	}{
		{"nil_input", nil, freq, rows, 0, heads, dim, half},
		{"nil_frequency", x, nil, rows, 0, heads, dim, half},
		{"zero_input_pointer", &Buffer{Size: x.Size}, freq, rows, 0, heads, dim, half},
		{"short_input", &Buffer{Ptr: x.Ptr, Size: x.Size - 4}, freq, rows, 0, heads, dim, half},
		{"short_frequency", x, &Buffer{Ptr: freq.Ptr, Size: freq.Size - 4}, rows, 0, heads, dim, half},
		{"zero_rows", x, freq, 0, 0, heads, dim, half},
		{"negative_position", x, freq, rows, -1, heads, dim, half},
		{"position_overflow", x, freq, rows, maxInt, heads, dim, half},
		{"u32_position_overflow", x, freq, rows, math.MaxUint32, heads, dim, half},
		{"u32_rows_overflow", x, freq, maxInt, 0, heads, dim, half},
		{"u32_element_overflow", x, freq, 1, 0, 65536, 65536, 1},
		{"zero_heads", x, freq, rows, 0, 0, dim, half},
		{"zero_dimension", x, freq, rows, 0, heads, 0, half},
		{"bad_rotary_width", x, freq, rows, 0, heads, dim, dim},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := RoPEPartialSequenceBuffer(tc.x, tc.freq, tc.rows, tc.pos0, tc.heads, tc.dim, tc.half); err == nil {
				t.Fatal("accepted invalid sequence RoPE dimensions or buffer")
			}
		})
	}
}
