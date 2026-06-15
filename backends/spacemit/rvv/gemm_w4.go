//go:build riscv64

package rvv

import (
	"sync"
	"unsafe"
)

//go:noescape
func kernelM4N32W4(a, b4 *int8, c *int32, K, lda, ldc int64)

// PackBW4 packs int4 weights W[N,K] (values in [-8,7]) into [N/32][K][16] tiles:
// byte j holds col j in the low nibble and col j+16 in the high nibble.
// Requires N%32==0. Half the bytes of the int8 PackB.
func PackBW4(W []int8, N, K int) []int8 {
	out := make([]int8, N/32*K*16)
	for nt := 0; nt < N/32; nt++ {
		base := nt * K * 16
		for k := 0; k < K; k++ {
			for j := 0; j < 16; j++ {
				lo := W[(nt*32+j)*K+k] & 0xF
				hi := W[(nt*32+16+j)*K+k] & 0xF
				out[base+k*16+j] = lo | (hi << 4)
			}
		}
	}
	return out
}

// GemmU8W4 computes raw int32 = aq[M,K] (uint8) · W4 (packed int4), threaded.
func GemmU8W4(aq []uint8, B4 []int8, C []int32, M, N, K, nthreads int) {
	mblocks := M / 4
	work := func(mb0, mb1 int) {
		for mb := mb0; mb < mb1; mb++ {
			m := mb * 4
			for nt := 0; nt < N/32; nt++ {
				kernelM4N32W4((*int8)(unsafe.Pointer(&aq[m*K])), &B4[nt*K*16],
					&C[m*N+nt*32], int64(K), int64(K), int64(N*4))
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
