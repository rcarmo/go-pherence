package rvv

import "sync"

// dotF16 returns sum(float32(a[i]) * float32(b[i])) for fp16 vectors stored as
// IEEE 754 binary16 words. Accumulation is float32 via RVV/Zvfh widening FMA.
//
//go:noescape
func dotF16(a, b *uint16, n int64) float32

//go:noescape
func kernelF16M4N16(a, bp *uint16, c *float32, K, lda, ldc int64)

//go:noescape
func kernelF16M4N32(a, bp *uint16, c *float32, K, lda, ldc int64)

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

// PackBF16 packs B[N,K] (transposed-B, row n = output n's fp16 weights) into
// tiles of 16 N-columns: Bp[nt][k][0:16]. Requires N % 16 == 0. Pre-pack static
// weights before calling GemmF16Outer.
func PackBF16(B []uint16, N, K int) []uint16 { return packBF16Tile(B, N, K, 16) }

// PackBF16N32 packs B[N,K] into tiles of 32 N-columns for GemmF16Outer32.
// It is the preferred layout for X100 VLEN=256 because it fills e16,m2/e32,m4.
func PackBF16N32(B []uint16, N, K int) []uint16 { return packBF16Tile(B, N, K, 32) }

func packBF16Tile(B []uint16, N, K, tileN int) []uint16 {
	Bp := make([]uint16, N*K)
	for nt := 0; nt < N/tileN; nt++ {
		base := nt * K * tileN
		for k := 0; k < K; k++ {
			for j := 0; j < tileN; j++ {
				Bp[base+k*tileN+j] = B[(nt*tileN+j)*K+k]
			}
		}
	}
	return Bp
}

// GemmF16Outer computes C[M,N] = A[M,K]·B[N,K]^T using the FP16 M4xN16
// outer-product kernel. Bp must come from PackBF16. Requires M%4==0 and N%16==0.
// nthreads partitions work over M-blocks.
func GemmF16Outer(A, Bp []uint16, C []float32, M, N, K, nthreads int) {
	gemmF16OuterTile(A, Bp, C, M, N, K, nthreads, 16, kernelF16M4N16)
}

// GemmF16Outer32 is the M4xN32 FP16 outer-product kernel. Bp must come from
// PackBF16N32. Requires M%4==0 and N%32==0.
func GemmF16Outer32(A, Bp []uint16, C []float32, M, N, K, nthreads int) {
	gemmF16OuterTile(A, Bp, C, M, N, K, nthreads, 32, kernelF16M4N32)
}

func gemmF16OuterTile(A, Bp []uint16, C []float32, M, N, K, nthreads, tileN int, kernel func(a, bp *uint16, c *float32, K, lda, ldc int64)) {
	mblocks := M / 4
	work := func(mb0, mb1 int) {
		for mb := mb0; mb < mb1; mb++ {
			m := mb * 4
			for nt := 0; nt < N/tileN; nt++ {
				kernel(&A[m*K], &Bp[nt*K*tileN], &C[m*N+nt*tileN],
					int64(K), int64(K*2), int64(N*4))
			}
		}
	}
	if nthreads <= 1 {
		work(0, mblocks)
		return
	}
	var wg sync.WaitGroup
	ch := (mblocks + nthreads - 1) / nthreads
	for t := 0; t < nthreads; t++ {
		m0, m1 := t*ch, (t+1)*ch
		if m0 >= mblocks {
			break
		}
		if m1 > mblocks {
			m1 = mblocks
		}
		wg.Add(1)
		go func(m0, m1 int) { defer wg.Done(); work(m0, m1) }(m0, m1)
	}
	wg.Wait()
}

// GemmF16Threaded runs GemmF16 across nthreads goroutines partitioned over M
// rows. It dispatches to the tiled M4xN32 kernel when dimensions are compatible,
// falls back to M4xN16, and finally to the dot-loop kernel for tails/small odd
// shapes.
func GemmF16Threaded(A, B []uint16, C []float32, M, N, K, nthreads int) {
	if M%4 == 0 && N%32 == 0 {
		GemmF16Outer32(A, PackBF16N32(B, N, K), C, M, N, K, nthreads)
		return
	}
	if M%4 == 0 && N%16 == 0 {
		GemmF16Outer(A, PackBF16(B, N, K), C, M, N, K, nthreads)
		return
	}

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
