//go:build riscv64

package simd

import (
	"math/rand"
	"testing"
	"time"
	"unsafe"
)

// perCellSgemmNT is the previous implementation: one RVV dot product per output
// cell. Kept here to A/B against the register-blocked 1x4 microkernel.
func perCellSgemmNT(m, n, k int, alpha float32, a, b, c unsafe.Pointer, lda, ldb, ldc int) {
	for i := 0; i < m; i++ {
		aRow := unsafe.Slice((*float32)(unsafe.Add(a, uintptr(i*lda)*4)), k)
		for j := 0; j < n; j++ {
			bRow := unsafe.Slice((*float32)(unsafe.Add(b, uintptr(j*ldb)*4)), k)
			sum := sdotAsm(aRow, bRow)
			storeF32(c, i*ldc+j, loadF32(c, i*ldc+j)+alpha*sum)
		}
	}
}

func TestSgemmNTKernelSpeed(t *testing.T) {
	if !HasSgemmAsm {
		t.Skip("no RVV SGEMM")
	}
	m, n, k := 256, 1280, 1280 // representative whisper encoder GEMM tile
	rng := rand.New(rand.NewSource(1))
	a := make([]float32, m*k)
	b := make([]float32, n*k)
	for i := range a {
		a[i] = rng.Float32()
	}
	for i := range b {
		b[i] = rng.Float32()
	}
	c1 := make([]float32, m*n)
	c2 := make([]float32, m*n)

	const reps = 3
	// warm up both
	SgemmNT(m, n, k, 1, unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&c1[0]), k, k, n)
	sgemmNT1x4(m, n, k, 1, unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&c2[0]), k, k, n)
	_ = perCellSgemmNT

	t0 := time.Now()
	for r := 0; r < reps; r++ {
		SgemmNT(m, n, k, 1, unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&c1[0]), k, k, n)
	}
	perCell := time.Since(t0) / reps

	t0 = time.Now()
	for r := 0; r < reps; r++ {
		sgemmNT1x4(m, n, k, 1, unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&c2[0]), k, k, n)
	}
	blocked := time.Since(t0) / reps

	flops := 2.0 * float64(m) * float64(n) * float64(k)
	t.Logf("per-cell (SgemmNT): %v  (%.2f GFLOP/s)", perCell, flops/perCell.Seconds()/1e9)
	t.Logf("blocked 1x4:        %v  (%.2f GFLOP/s)", blocked, flops/blocked.Seconds()/1e9)
	t.Logf("blocked/per-cell:   %.2fx", float64(perCell)/float64(blocked))
}
