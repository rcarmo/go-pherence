//go:build riscv64

package ime2

import "unsafe"

// vmadotKLoop runs the inner K-loop in assembly for pre-packed tiles.
// A and B must be pre-packed in 4×8 tile layout: K/8 consecutive 32-byte tiles.
// C is a 4×4 int32 accumulator (64 bytes, loaded/stored by the asm).
// K must be a multiple of 8.
//
//go:noescape
func vmadotKLoop(A *byte, B *byte, C *int32, K int)

// PackTiles converts a row-major INT8 matrix [rows × K] into
// pre-packed 4×8 tile layout for vmadot consumption.
// Output layout: for each group of 4 rows, K/8 tiles of 32 bytes each.
// Total output size: rows * K bytes (same as input, just reordered).
func PackTiles(src []int8, rows, K int) []int8 {
	if rows%4 != 0 || K%8 != 0 {
		panic("ime2: PackTiles requires rows%4==0, K%8==0")
	}
	dst := make([]int8, rows*K)
	for rg := 0; rg < rows; rg += 4 {
		for ki := 0; ki < K; ki += 8 {
			tileIdx := (rg/4)*(K/8) + ki/8
			tileBase := tileIdx * 32
			for r := 0; r < 4; r++ {
				copy(dst[tileBase+r*8:tileBase+r*8+8],
					src[(rg+r)*K+ki:(rg+r)*K+ki+8])
			}
		}
	}
	return dst
}

// GemmINT8Packed performs C[M×N] = A_packed * B_packed^T using vmadot.
// A_packed: M/4 row-groups × K/8 tiles of 32 bytes (call PackTiles on A first)
// B_packed: N/4 row-groups × K/8 tiles of 32 bytes (call PackTiles on B first)
// C: output [M×N] int32 in row-major.
// M, N must be multiples of 4. K must be a multiple of 8.
func GemmINT8Packed(M, N, K int, Apacked, Bpacked []int8, C []int32) {
	if M%4 != 0 || N%4 != 0 || K%8 != 0 {
		panic("ime2: dimensions must be multiples of 4/4/8")
	}
	tilesPerRow := K / 8 // number of 32-byte tiles per row-group

	for i := 0; i < M; i += 4 {
		aBase := (i / 4) * tilesPerRow * 32
		for j := 0; j < N; j += 4 {
			bBase := (j / 4) * tilesPerRow * 32

			var acc [16]int32
			vmadotKLoop(
				(*byte)(unsafe.Pointer(&Apacked[aBase])),
				(*byte)(unsafe.Pointer(&Bpacked[bBase])),
				&acc[0],
				K,
			)

			// Write output tile
			for r := 0; r < 4; r++ {
				for c := 0; c < 4; c++ {
					C[(i+r)*N+(j+c)] = acc[r*4+c]
				}
			}
		}
	}
}

// PackTilesInto packs tiles into a pre-allocated destination buffer.
// dst must have at least rows*K bytes capacity.
func PackTilesInto(src []int8, rows, K int, dst []int8) []int8 {
	if rows%4 != 0 || K%8 != 0 {
		panic("ime2: PackTilesInto requires rows%4==0, K%8==0")
	}
	needed := rows * K
	out := dst[:needed]
	for rg := 0; rg < rows; rg += 4 {
		for ki := 0; ki < K; ki += 8 {
			tileIdx := (rg/4)*(K/8) + ki/8
			tileBase := tileIdx * 32
			for r := 0; r < 4; r++ {
				copy(out[tileBase+r*8:tileBase+r*8+8],
					src[(rg+r)*K+ki:(rg+r)*K+ki+8])
			}
		}
	}
	return out
}

// VmadotKLoop is the exported version of vmadotKLoop.


func VmadotKLoop(A *byte, B *byte, C *int32, K int) { vmadotKLoop(A, B, C, K) }

