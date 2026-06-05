//go:build riscv64

package simd

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"
)

// Reference C += alpha * A * B^T (row-major, contiguous rows).
func refSgemmNT(c, a, b []float32, m, n, k int, alpha float32, lda, ldb, ldc int) {
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var s float32
			for p := 0; p < k; p++ {
				s += a[i*lda+p] * b[j*ldb+p]
			}
			c[i*ldc+j] += alpha * s
		}
	}
}

func TestSgemmNTKernelMatchesReference(t *testing.T) {
	if !HasSgemmAsm {
		t.Skip("no RVV SGEMM on this host")
	}
	rng := rand.New(rand.NewSource(7))
	cases := []struct{ m, n, k int }{
		{1, 1, 1}, {1, 4, 8}, {3, 5, 17}, {7, 9, 33},
		{4, 4, 64}, {2, 11, 128}, {1, 13, 257}, {1500, 4, 80},
		{8, 1280, 1280},
	}
	for _, tc := range cases {
		m, n, k := tc.m, tc.n, tc.k
		a := make([]float32, m*k)
		b := make([]float32, n*k)
		for i := range a {
			a[i] = rng.Float32()*2 - 1
		}
		for i := range b {
			b[i] = rng.Float32()*2 - 1
		}
		got := make([]float32, m*n)
		gotBlk := make([]float32, m*n)
		alpha := float32(1.0)
		SgemmNT(m, n, k, alpha, unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&got[0]), k, k, n)
		sgemmNT1x4(m, n, k, alpha, unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&gotBlk[0]), k, k, n)
		// Compare both against a float64 truth; f32 reassociation differs, so
		// judge on absolute error scaled by the f32 accumulation bound.
		var maxAbs, maxAbsBlk float64
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				var t64 float64
				for p := 0; p < k; p++ {
					t64 += float64(a[i*k+p]) * float64(b[j*k+p])
				}
				if d := math.Abs(float64(got[i*n+j]) - t64); d > maxAbs {
					maxAbs = d
				}
				if d := math.Abs(float64(gotBlk[i*n+j]) - t64); d > maxAbsBlk {
					maxAbsBlk = d
				}
			}
		}
		tol := 2e-3 * float64(k) * 1.2e-7 * 16 // generous f32 error budget
		if tol < 1e-3 {
			tol = 1e-3
		}
		if maxAbs > tol {
			t.Fatalf("SgemmNT m=%d n=%d k=%d maxAbsErr=%g tol=%g", m, n, k, maxAbs, tol)
		}
		if maxAbsBlk > tol {
			t.Fatalf("sgemmNT1x4 m=%d n=%d k=%d maxAbsErr=%g tol=%g", m, n, k, maxAbsBlk, tol)
		}
	}
}
