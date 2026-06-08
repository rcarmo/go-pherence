package rvv

import "sync"

// dotF16 returns sum(float32(a[i]) * float32(b[i])) for fp16 vectors stored as
// IEEE 754 binary16 words. Accumulation is float32 via RVV/Zvfh widening FMA.
//
//go:noescape
func dotF16(a, b *uint16, n int64) float32

// DotF16 returns the dot product of two fp16 vectors, accumulated in float32.
// The inputs are IEEE 754 binary16 words. It is the FP16/Zvfh counterpart to
// DotF32RVV/dotI8 and is the first building block for K3-native fp16 attention
// and GEMM kernels.
func DotF16(a, b []uint16) float32 {
	if len(a) == 0 {
		return 0
	}
	if len(b) < len(a) {
		panic("rvv.DotF16: len(b) < len(a)")
	}
	return dotF16(&a[0], &b[0], int64(len(a)))
}

// GemmF16 computes C[M,N] = A[M,K] · B[N,K]^T with fp16 inputs and fp32
// accumulation/output. A is row-major [M,K], B is row-major [N,K] (transposed-B,
// one row per output channel), and C is row-major [M,N]. This mirrors the int8
// GemmI8 contract and the EP's spe_mmt4d_transb semantics while using Zvfh
// widening FMA (f16*f16 -> f32).
func GemmF16(A, B []uint16, C []float32, M, N, K int) {
	for m := 0; m < M; m++ {
		ar := &A[m*K]
		cr := C[m*N : m*N+N]
		for n := 0; n < N; n++ {
			cr[n] = dotF16(ar, &B[n*K], int64(K))
		}
	}
}

// GemmF16Threaded runs GemmF16 across nthreads goroutines partitioned over M
// rows. This first implementation intentionally reuses the proven DotF16 kernel;
// future m4n/tiled kernels can preserve the same public contract.
func GemmF16Threaded(A, B []uint16, C []float32, M, N, K, nthreads int) {
	if nthreads <= 1 {
		GemmF16(A, B, C, M, N, K)
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
					cr[n] = dotF16(ar, &B[n*K], int64(K))
				}
			}
		}(m0, m1)
	}
	wg.Wait()
}
