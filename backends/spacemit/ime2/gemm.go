package ime2

import (
	"unsafe"
)

// GemmINT8 performs C[M×N] += A[M×K] * B[K×N] using vmadot.
// A and B are pre-quantized INT8 matrices in row-major order.
// C is INT32 accumulator.
// M, N must be multiples of 4. K must be a multiple of 8.
func GemmINT8(M, N, K int, A, B []int8, C []int32) {
	// vmadot computes: C_tile[4×4] += A_tile[4×8] × B_tile[4×8]^T
	// So we tile: M in steps of 4, N in steps of 4, K in steps of 8
	
	for i := 0; i < M; i += 4 {
		for j := 0; j < N; j += 4 {
			// Accumulator for this 4x4 output tile
			var acc [16]int32 // 4×4 in row-major

			for k := 0; k < K; k += 8 {
				// A tile: rows i..i+3, cols k..k+7 → 4×8 contiguous chunk
				// But A is row-major [M×K], so row r starts at r*K
				var aTile [32]byte
				for r := 0; r < 4; r++ {
					for c := 0; c < 8; c++ {
						aTile[r*8+c] = byte(A[(i+r)*K+(k+c)])
					}
				}

				// B tile: rows j..j+3, cols k..k+7 → 4×8 chunk
				// B is [N×K] (transposed: each row of B is a column of the original)
				var bTile [32]byte
				for r := 0; r < 4; r++ {
					for c := 0; c < 8; c++ {
						bTile[r*8+c] = byte(B[(j+r)*K+(k+c)])
					}
				}

				// vmadot: acc += aTile * bTile^T
				vmadotAccSS4x8(&aTile[0], &bTile[0], &acc[0])
			}

			// Store accumulated results
			for r := 0; r < 4; r++ {
				for c := 0; c < 4; c++ {
					C[(i+r)*N+(j+c)] += acc[r*4+c]
				}
			}
		}
	}
}

// GemmINT8Simple performs C[M×N] = A[M×K] * B[N×K]^T using vmadot.
// B is stored in transposed layout [N×K] for cache efficiency.
// All dimensions must be multiples of 4 (M,N) and 8 (K).
func GemmINT8Simple(M, N, K int, A []int8, B []int8, C []int32) {
	if M%4 != 0 || N%4 != 0 || K%8 != 0 {
		panic("ime2: dimensions must be multiples of 4/4/8")
	}

	for i := 0; i < M; i += 4 {
		for j := 0; j < N; j += 4 {
			var acc [16]int32

			for k := 0; k < K; k += 8 {
				var aTile, bTile [32]byte
				
				// Pack A tile (4 rows from A, 8 consecutive K elements)
				aBase := i*K + k
				copy(aTile[0:8], (*[8]byte)(unsafe.Pointer(&A[aBase]))[:])
				copy(aTile[8:16], (*[8]byte)(unsafe.Pointer(&A[aBase+K]))[:])
				copy(aTile[16:24], (*[8]byte)(unsafe.Pointer(&A[aBase+2*K]))[:])
				copy(aTile[24:32], (*[8]byte)(unsafe.Pointer(&A[aBase+3*K]))[:])

				// Pack B tile (4 rows from B, 8 consecutive K elements)
				bBase := j*K + k
				copy(bTile[0:8], (*[8]byte)(unsafe.Pointer(&B[bBase]))[:])
				copy(bTile[8:16], (*[8]byte)(unsafe.Pointer(&B[bBase+K]))[:])
				copy(bTile[16:24], (*[8]byte)(unsafe.Pointer(&B[bBase+2*K]))[:])
				copy(bTile[24:32], (*[8]byte)(unsafe.Pointer(&B[bBase+3*K]))[:])

				vmadotAccSS4x8(&aTile[0], &bTile[0], &acc[0])
			}

			// Write output
			for r := 0; r < 4; r++ {
				for c := 0; c < 4; c++ {
					C[(i+r)*N+(j+c)] = acc[r*4+c]
				}
			}
		}
	}
}
