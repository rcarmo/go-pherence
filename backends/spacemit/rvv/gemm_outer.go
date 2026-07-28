//go:build riscv64

package rvv

import "sync"

//go:noescape
func kernelM4N32(a, bp *int8, c *int32, K, lda, ldc int64)

// PackB packs B[N,K] (transposed-B, row n = output n's weights) into tiles of
// 32 N-columns: Bp[nt][k][0:32]. The final partial tile is zero-padded so the
// packed layout remains safe for arbitrary N. Pre-pack static weights.
func PackB(B []int8, N, K int) []int8 { return packI8TilePadded(B, N, K, 32) }

// GemmI8Outer computes C[M,N] = A[M,K]·B[N,K]^T using the outer-product kernel.
// Bp must come from PackB. The M4xN32 assembly core is kept for full tiles;
// arbitrary M/N tails are handled with a scratch tile and scalar packed fallback.
// Work is partitioned over full M-blocks. K-blocking is intentionally omitted
// because the assembly kernel overwrites C rather than accumulating into it.
func GemmI8Outer(A, Bp []int8, C []int32, M, N, K, nthreads int) {
	const tileN = 32
	mblocks := M / 4
	fullNTiles := N / tileN
	tailN := N - fullNTiles*tileN
	work := func(mb0, mb1 int) {
		var tailTile [4 * tileN]int32
		for nb0 := 0; nb0 < fullNTiles; nb0 += outerCacheBlockTiles {
			nb1 := minInt(fullNTiles, nb0+outerCacheBlockTiles)
			for mb := mb0; mb < mb1; mb++ {
				m := mb * 4
				for nt := nb0; nt < nb1; nt++ {
					kernelM4N32(&A[m*K], &Bp[nt*K*tileN], &C[m*N+nt*tileN],
						int64(K), int64(K), int64(N*4))
				}
			}
		}
		if tailN == 0 {
			return
		}
		for mb := mb0; mb < mb1; mb++ {
			m := mb * 4
			kernelM4N32(&A[m*K], &Bp[fullNTiles*K*tileN], &tailTile[0],
				int64(K), int64(K), int64(tileN*4))
			copyI32TailTile(C[m*N+fullNTiles*tileN:], N, tailN, tileN, tailTile[:])
		}
	}
	if nthreads <= 1 {
		work(0, mblocks)
	} else {
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
	if mblocks*4 < M {
		gemmI8PackedRowsScalar(A, Bp, C, mblocks*4, M, N, K, tileN)
	}
}
