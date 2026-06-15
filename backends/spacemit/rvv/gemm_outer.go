//go:build riscv64

package rvv

import "sync"

//go:noescape
func kernelM4N32(a, bp *int8, c *int32, K, lda, ldc int64)

// PackB packs B[N,K] (transposed-B, row n = output n's weights) into tiles of
// 32 N-columns: Bp[nt][k][0:32]. Requires N % 32 == 0. Pre-pack static weights.
func PackB(B []int8, N, K int) []int8 {
	Bp := make([]int8, N*K)
	for nt := 0; nt < N/32; nt++ {
		base := nt * K * 32
		for k := 0; k < K; k++ {
			for j := 0; j < 32; j++ {
				Bp[base+k*32+j] = B[(nt*32+j)*K+k]
			}
		}
	}
	return Bp
}

// GemmI8Outer computes C[M,N] = A[M,K]·B[N,K]^T using the outer-product kernel.
// Bp must come from PackB. Requires M%4==0 and N%32==0. nthreads over M-blocks.
func GemmI8Outer(A, Bp []int8, C []int32, M, N, K, nthreads int) {
	mblocks := M / 4
	work := func(mb0, mb1 int) {
		for mb := mb0; mb < mb1; mb++ {
			m := mb * 4
			for nt := 0; nt < N/32; nt++ {
				kernelM4N32(&A[m*K], &Bp[nt*K*32], &C[m*N+nt*32],
					int64(K), int64(K), int64(N*4))
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
		a, b := t*ch, (t+1)*ch
		if a >= mblocks {
			break
		}
		if b > mblocks {
			b = mblocks
		}
		wg.Add(1)
		go func(a, b int) { defer wg.Done(); work(a, b) }(a, b)
	}
	wg.Wait()
}
