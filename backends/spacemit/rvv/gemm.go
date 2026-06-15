//go:build riscv64

// Package rvv hosts pure-Go (Go asm, no cgo) RVV int8 microkernels for the
// SpaceMIT K3 X60 cores. The "NPU" is RVV; these kernels reimplement the EP's
// spe_mmt4d_transb path. Vector instructions are WORD-encoded in the .s files.
package rvv

import "sync"

// dotI8 returns the int32 dot product of two int8 vectors of length n.
//
//go:noescape
func dotI8(a, b *int8, n int64) int32

// GemmI8 computes C[M,N] = A[M,K] · B[N,K]^T (transposed B), int8->int32.
// A is row-major [M,K], B is row-major [N,K] (each row is one output's weights),
// C is row-major [M,N]. This mirrors the EP's spe_mmt4d_transb semantics.
func GemmI8(A, B []int8, C []int32, M, N, K int) {
	for m := 0; m < M; m++ {
		ar := &A[m*K]
		cr := C[m*N : m*N+N]
		for n := 0; n < N; n++ {
			cr[n] = dotI8(ar, &B[n*K], int64(K))
		}
	}
}

// GemmI8Threaded runs GemmI8 across nthreads goroutines partitioned over M rows.
func GemmI8Threaded(A, B []int8, C []int32, M, N, K, nthreads int) {
	if nthreads <= 1 {
		GemmI8(A, B, C, M, N, K)
		return
	}
	var wg sync.WaitGroup
	chunk := (M + nthreads - 1) / nthreads
	for t := 0; t < nthreads; t++ {
		m0 := t * chunk
		if m0 >= M {
			break
		}
		m1 := m0 + chunk
		if m1 > M {
			m1 = M
		}
		wg.Add(1)
		go func(m0, m1 int) {
			defer wg.Done()
			for m := m0; m < m1; m++ {
				ar := &A[m*K]
				cr := C[m*N : m*N+N]
				for n := 0; n < N; n++ {
					cr[n] = dotI8(ar, &B[n*K], int64(K))
				}
			}
		}(m0, m1)
	}
	wg.Wait()
}
