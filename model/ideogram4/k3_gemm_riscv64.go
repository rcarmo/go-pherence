//go:build riscv64

package ideogram4

import (
	"github.com/rcarmo/go-pherence/backends/spacemit/rvv"
)

func k3GemmRowsF32(out []float32, a, b []float32, m, n, k int) bool {
	if !k3Enabled() || m <= 0 || n <= 0 || k <= 0 || len(out) < m*n || len(a) < m*k || len(b) < n*k {
		return false
	}
	A := make([]uint16, m*k)
	B := make([]uint16, n*k)
	rvv.F32ToF16RVV(A, a[:m*k])
	rvv.F32ToF16RVV(B, b[:n*k])
	rvv.GemmF16Threaded(A, B, out[:m*n], m, n, k, k3Threads())
	return true
}
