package ime2

import "unsafe"

// GemmINT8Direct performs C[M×N] = A[M×K] * B[N×K]^T using vmadot.
// Same as GemmINT8Simple but with minor optimizations.
// Requires M,N multiples of 4, K multiple of 8.
func GemmINT8Direct(M, N, K int, A []int8, B []int8, C []int32) {
	if M%4 != 0 || N%4 != 0 || K%8 != 0 {
		panic("ime2: dimensions must be multiples of 4/4/8")
	}

	for i := 0; i < M; i += 4 {
		for j := 0; j < N; j += 4 {
			var acc [16]int32

			for k := 0; k < K; k += 8 {
				var aTile, bTile [32]byte
				for r := 0; r < 4; r++ {
					*(*[8]byte)(unsafe.Pointer(&aTile[r*8])) = *(*[8]byte)(unsafe.Pointer(&A[(i+r)*K+k]))
				}
				for r := 0; r < 4; r++ {
					*(*[8]byte)(unsafe.Pointer(&bTile[r*8])) = *(*[8]byte)(unsafe.Pointer(&B[(j+r)*K+k]))
				}
				vmadotAccSS4x8(&aTile[0], &bTile[0], &acc[0])
			}

			for r := 0; r < 4; r++ {
				for c := 0; c < 4; c++ {
					C[(i+r)*N+(j+c)] = acc[r*4+c]
				}
			}
		}
	}
}
