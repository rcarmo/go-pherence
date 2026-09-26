package simd

import (
	"slices"
	"testing"
	"unsafe"
)

// Offset K-block views can end inside the final row. Run with -race to enable
// checkptr; an unsafe.Slice sized as m*lda would straddle the allocation.
func TestGebpOffsetReductionViewBounds(t *testing.T) {
	if !HasSgemmAsm {
		t.Skip("GEBP assembly unavailable")
	}
	const m, n, k, stride, offset = 7, 19, 4, 11, 7
	a := make([]float32, m*stride)
	b := make([]float32, n*stride)
	c := make([]float32, m*n)
	for i := range a {
		a[i] = float32(i%13-6) / 8
	}
	for i := range b {
		b[i] = float32(i%11-5) / 8
	}
	want := slices.Clone(c)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var v float32
			for p := 0; p < k; p++ {
				v += a[i*stride+offset+p] * b[j*stride+offset+p]
			}
			want[i*n+j] = v
		}
	}
	pack := make([]float32, k*gebpNR)
	if SgemmNTGebpWithPack(m, n, k, 1, unsafe.Pointer(&a[offset]), unsafe.Pointer(&b[offset]), unsafe.Pointer(&c[0]), stride, stride, n, pack[:len(pack)-1]) {
		t.Fatal("accepted short packing scratch")
	}
	if SgemmNTGebpWithPack(m, n, k, 1, unsafe.Pointer(&a[offset]), unsafe.Pointer(&b[offset]), unsafe.Pointer(&c[0]), stride, stride, n, a[:len(pack)]) {
		t.Fatal("accepted overlapping packing scratch")
	}
	if !SgemmNTGebpWithPack(m, n, k, 1, unsafe.Pointer(&a[offset]), unsafe.Pointer(&b[offset]), unsafe.Pointer(&c[0]), stride, stride, n, pack) {
		t.Fatal("rejected offset reduction view")
	}
	for i := range c {
		if d := c[i] - want[i]; d > 1e-5 || d < -1e-5 {
			t.Fatalf("index%d got%g want%g", i, c[i], want[i])
		}
	}
	if allocs := testing.AllocsPerRun(10, func() {
		clear(c)
		if !SgemmNTGebpWithPack(m, n, k, 1, unsafe.Pointer(&a[offset]), unsafe.Pointer(&b[offset]), unsafe.Pointer(&c[0]), stride, stride, n, pack) {
			panic("valid GEMM rejected")
		}
	}); allocs != 0 {
		t.Fatalf("caller-owned GEMM pack allocated %g times", allocs)
	}
}
