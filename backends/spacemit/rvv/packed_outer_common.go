package rvv

import "github.com/rcarmo/go-pherence/half"

const outerCacheBlockTiles = 2

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func packedTileCount(N, tileN int) int {
	if N <= 0 || tileN <= 0 {
		return 0
	}
	return (N + tileN - 1) / tileN
}

func packedTileLen(N, K, tileN int) int {
	return packedTileCount(N, tileN) * K * tileN
}

func packI8TilePadded(B []int8, N, K, tileN int) []int8 {
	return packI8TilePaddedInto(B, N, K, tileN, make([]int8, packedTileLen(N, K, tileN)))
}

func packI8TilePaddedInto(B []int8, N, K, tileN int, dst []int8) []int8 {
	bpLen := packedTileLen(N, K, tileN)
	if cap(dst) < bpLen {
		panic("rvv: cap(dst) < packed int8 tile length")
	}
	Bp := dst[:bpLen]
	for nt := 0; nt < packedTileCount(N, tileN); nt++ {
		cols := minInt(tileN, N-nt*tileN)
		base := nt * K * tileN
		for k := 0; k < K; k++ {
			row := Bp[base+k*tileN : base+(k+1)*tileN]
			if cols < tileN {
				clear(row[cols:])
			}
			for j := 0; j < cols; j++ {
				row[j] = B[(nt*tileN+j)*K+k]
			}
		}
	}
	return Bp
}

func packBF16TilePadded(B []uint16, N, K, tileN int) []uint16 {
	return packBF16TilePaddedInto(B, N, K, tileN, make([]uint16, packedTileLen(N, K, tileN)))
}

func packBF16TilePaddedInto(B []uint16, N, K, tileN int, dst []uint16) []uint16 {
	bpLen := packedTileLen(N, K, tileN)
	if cap(dst) < bpLen {
		panic("rvv: cap(dst) < packed fp16 tile length")
	}
	Bp := dst[:bpLen]
	for nt := 0; nt < packedTileCount(N, tileN); nt++ {
		cols := minInt(tileN, N-nt*tileN)
		base := nt * K * tileN
		for k := 0; k < K; k++ {
			row := Bp[base+k*tileN : base+(k+1)*tileN]
			if cols < tileN {
				clear(row[cols:])
			}
			for j := 0; j < cols; j++ {
				row[j] = B[(nt*tileN+j)*K+k]
			}
		}
	}
	return Bp
}

func gemmI8PackedRowsScalar(A, Bp []int8, C []int32, m0, m1, N, K, tileN int) {
	for m := m0; m < m1; m++ {
		ar := A[m*K : m*K+K]
		cr := C[m*N : m*N+N]
		for n := 0; n < N; n++ {
			nt, j := n/tileN, n%tileN
			base := nt * K * tileN
			var sum int32
			for k := 0; k < K; k++ {
				sum += int32(ar[k]) * int32(Bp[base+k*tileN+j])
			}
			cr[n] = sum
		}
	}
}

func gemmF16PackedRowsScalar(A, Bp []uint16, C []float32, m0, m1, N, K, tileN int) {
	for m := m0; m < m1; m++ {
		ar := A[m*K : m*K+K]
		cr := C[m*N : m*N+N]
		for n := 0; n < N; n++ {
			nt, j := n/tileN, n%tileN
			base := nt * K * tileN
			var sum float32
			for k := 0; k < K; k++ {
				sum += half.F16ToF32(ar[k]) * half.F16ToF32(Bp[base+k*tileN+j])
			}
			cr[n] = sum
		}
	}
}

func copyI32TailTile(dst []int32, dstStride, cols, tileN int, src []int32) {
	for r := 0; r < 4; r++ {
		copy(dst[r*dstStride:r*dstStride+cols], src[r*tileN:r*tileN+cols])
	}
}

func copyF32TailTile(dst []float32, dstStride, cols, tileN int, src []float32) {
	for r := 0; r < 4; r++ {
		copy(dst[r*dstStride:r*dstStride+cols], src[r*tileN:r*tileN+cols])
	}
}
