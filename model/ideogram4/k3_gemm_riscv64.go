//go:build riscv64

package ideogram4

import (
	"sync"
	"unsafe"

	"github.com/rcarmo/go-pherence/backends/spacemit/ime2"
	"github.com/rcarmo/go-pherence/backends/spacemit/aicpu/aipool"
	"github.com/rcarmo/go-pherence/backends/spacemit/rvv"
)

var k3GemmWeightCache sync.Map // uintptr(&b[0]) -> ime2.Q80x32

func k3GetCachedQ80(b []float32, n, k int) (ime2.Q80x32, bool) {
	key := uintptr(unsafe.Pointer(&b[0]))
	if v, ok := k3GemmWeightCache.Load(key); ok {
		return v.(ime2.Q80x32), true
	}
	q := ime2.PackF32ToQ80x32RowScale(n, k, b)
	if !q.Valid {
		return q, false
	}
	k3GemmWeightCache.Store(key, q)
	return q, true
}

func k3GemmRowsF32(out []float32, a, b []float32, m, n, k int) bool {
	if !k3Enabled() || m <= 0 || n <= 0 || k <= 0 || len(out) < m*n || len(a) < m*k || len(b) < n*k {
		return false
	}
	// Try A100 Q8 path for large aligned shapes (VAE convs, large projections).
	if k3A100Q8Enabled() && m >= 4 && n%32 == 0 && k%32 == 0 {
		wq, ok := k3GetCachedQ80(b[:n*k], n, k)
		if ok {
			if aipool.GemmQ80x32AIPooledX100Pack(a[:m*k], m, k, wq, out[:m*n], k3A100WorkerPool()) {
				return true
			}
		}
	}
	// Fallback: FP16 RVV GEMM
	A := make([]uint16, m*k)
	B := make([]uint16, n*k)
	rvv.F32ToF16RVV(A, a[:m*k])
	rvv.F32ToF16RVV(B, b[:n*k])
	rvv.GemmF16Threaded(A, B, out[:m*n], m, n, k, k3Threads())
	return true
}
