//go:build !riscv64

package ime2

import "unsafe"

func vmadotSS4x8(A *byte, B *byte, C *int32) {
	a := unsafe.Slice((*int8)(unsafe.Pointer(A)), 32)
	b := unsafe.Slice((*int8)(unsafe.Pointer(B)), 32)
	c := unsafe.Slice(C, 16)
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			var acc int32
			for k := 0; k < 8; k++ {
				acc += int32(a[i*8+k]) * int32(b[j*8+k])
			}
			c[i*4+j] += acc
		}
	}
}

func vmadotUS4x8(A *byte, B *byte, C *int32)    { vmadotSS4x8(A, B, C) }
func vmadotAccSS4x8(A *byte, B *byte, C *int32) { vmadotSS4x8(A, B, C) }

func vmadotKLoop(A *byte, B *byte, C *int32, K int) {
	a := unsafe.Slice((*int8)(unsafe.Pointer(A)), 4*K)
	b := unsafe.Slice((*int8)(unsafe.Pointer(B)), 4*K)
	c := unsafe.Slice(C, 16)
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			var acc int32
			for k := 0; k < K; k++ {
				acc += int32(a[i*K+k]) * int32(b[j*K+k])
			}
			c[i*4+j] += acc
		}
	}
}

func Matmul4x8(A, B []int8, C []int32) {
	if len(A) < 32 || len(B) < 32 || len(C) < 16 {
		panic("ime2: buffer too small")
	}
	vmadotSS4x8((*byte)(unsafe.Pointer(&A[0])), (*byte)(unsafe.Pointer(&B[0])), &C[0])
}

func VmadotAccSS4x8(A *byte, B *byte, C *int32) { vmadotAccSS4x8(A, B, C) }

func PackTiles(src []int8, rows, K int) []int8 {
	if rows%4 != 0 || K%8 != 0 {
		panic("ime2: PackTiles requires rows%4==0, K%8==0")
	}
	dst := make([]int8, rows*K)
	return PackTilesInto(src, rows, K, dst)
}

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
				copy(out[tileBase+r*8:tileBase+r*8+8], src[(rg+r)*K+ki:(rg+r)*K+ki+8])
			}
		}
	}
	return out
}

func GemmINT8Packed(M, N, K int, Apacked, Bpacked []int8, C []int32) {
	if M%4 != 0 || N%4 != 0 || K%8 != 0 {
		panic("ime2: dimensions must be multiples of 4/4/8")
	}
	tilesPerRow := K / 8
	for i := 0; i < M; i += 4 {
		aBase := (i / 4) * tilesPerRow * 32
		for j := 0; j < N; j += 4 {
			bBase := (j / 4) * tilesPerRow * 32
			var acc [16]int32
			vmadotKLoop((*byte)(unsafe.Pointer(&Apacked[aBase])), (*byte)(unsafe.Pointer(&Bpacked[bBase])), &acc[0], K)
			for r := 0; r < 4; r++ {
				for c := 0; c < 4; c++ {
					C[(i+r)*N+(j+c)] = acc[r*4+c]
				}
			}
		}
	}
}
