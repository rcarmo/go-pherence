package ime2

import "github.com/rcarmo/go-pherence/internal/checked"

// Validate full Go slice extents before handing pointers to native tile loops.
// Malformed dimensions retain the historical panic contract, before any write.
func packedSize(rows, k int) int {
	n, ok := checked.MulInt(rows, k)
	if !ok || rows%4 != 0 || k%8 != 0 {
		panic("ime2: invalid/overflowing packed dimensions")
	}
	return n
}
func validatePacked(M, N, K int, a, b []int8, c []int32) bool {
	an, bn := packedSize(M, K), packedSize(N, K)
	cn, ok := checked.MulInt(M, N)
	if !ok || len(a) < an || len(b) < bn || len(c) < cn {
		panic("ime2: short/overflowing packed buffers")
	}
	return M > 0 && N > 0 && K > 0
}
