//go:build riscv64

package rvv

import "sync"

// GemmF16Outer32Spec describes one prepacked M4xN32 FP16 GEMM for batched
// dispatch. Bp must come from PackBF16N32.
type GemmF16Outer32Spec struct {
	A, Bp   []uint16
	C       []float32
	M, N, K int
}

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
// tiles of 16 N-columns: Bp[nt][k][0:16]. The final partial tile is zero-padded
// so the packed layout remains safe for arbitrary N.
func PackBF16(B []uint16, N, K int) []uint16 { return packBF16Tile(B, N, K, 16) }

// PackBF16Into is the allocation-free form of PackBF16. dst must have capacity
// for ceil(N/16)*K*16 elements and is returned for call chaining.
func PackBF16Into(B []uint16, N, K int, dst []uint16) []uint16 {
	return packBF16TileInto(B, N, K, 16, dst)
}

// PackBF16N32 packs B[N,K] into tiles of 32 N-columns for GemmF16Outer32.
// It is the preferred layout for X100 VLEN=256 because it fills e16,m2/e32,m4;
// the final partial tile is zero-padded.
func PackBF16N32(B []uint16, N, K int) []uint16 { return packBF16Tile(B, N, K, 32) }

// PackBF16N32Into is the allocation-free form of PackBF16N32. dst must have
// capacity for ceil(N/32)*K*32 elements and is returned for call chaining.
func PackBF16N32Into(B []uint16, N, K int, dst []uint16) []uint16 {
	return packBF16TileInto(B, N, K, 32, dst)
}

func packBF16Tile(B []uint16, N, K, tileN int) []uint16 {
	return packBF16TilePadded(B, N, K, tileN)
}

func packBF16TileInto(B []uint16, N, K, tileN int, dst []uint16) []uint16 {
	return packBF16TilePaddedInto(B, N, K, tileN, dst)
}

// GemmF16Outer computes C[M,N] = A[M,K]·B[N,K]^T using the FP16 M4xN16
// outer-product kernel. Bp must come from PackBF16. Full tiles stay on the
// assembly core; arbitrary M/N tails are handled safely via a scratch tile and
// scalar packed fallback. nthreads partitions work over full M-blocks.
func GemmF16Outer(A, Bp []uint16, C []float32, M, N, K, nthreads int) {
	gemmF16OuterTile(A, Bp, C, M, N, K, nthreads, 16, kernelF16M4N16)
}

// GemmF16Outer32 is the M4xN32 FP16 outer-product kernel. Bp must come from
// PackBF16N32. It preserves the fast M4xN32 core while safely handling tails.
func GemmF16Outer32(A, Bp []uint16, C []float32, M, N, K, nthreads int) {
	gemmF16OuterTile(A, Bp, C, M, N, K, nthreads, 32, kernelF16M4N32)
}

// GemmF16Outer32Batch runs multiple independent M4xN32 FP16 GEMMs with a single
// worker fanout. Exact-tile batches stay on the flattened fast path; tail cases
// fall back to per-spec GemmF16Outer32 so correctness is preserved.
func GemmF16Outer32Batch(nthreads int, specs ...GemmF16Outer32Spec) {
	if len(specs) == 0 {
		return
	}
	if nthreads <= 1 {
		for _, sp := range specs {
			GemmF16Outer32(sp.A, sp.Bp, sp.C, sp.M, sp.N, sp.K, 1)
		}
		return
	}
	for _, sp := range specs {
		if sp.M%4 != 0 || sp.N%32 != 0 {
			for _, sp := range specs {
				GemmF16Outer32(sp.A, sp.Bp, sp.C, sp.M, sp.N, sp.K, nthreads)
			}
			return
		}
	}
	starts := make([]int, len(specs)+1)
	for i, sp := range specs {
		starts[i+1] = starts[i] + sp.M/4
	}
	total := starts[len(specs)]
	var wg sync.WaitGroup
	for wid := 0; wid < nthreads; wid++ {
		wg.Add(1)
		go func(wid int) {
			defer wg.Done()
			for flat := wid; flat < total; flat += nthreads {
				// len(specs) is small (heads per batch), so linear search is cheaper
				// than extra bookkeeping and keeps this helper allocation-light.
				si := 0
				for starts[si+1] <= flat {
					si++
				}
				sp := specs[si]
				mb := flat - starts[si]
				m := mb * 4
				for nt := 0; nt < sp.N/32; nt++ {
					kernelF16M4N32(&sp.A[m*sp.K], &sp.Bp[nt*sp.K*32], &sp.C[m*sp.N+nt*32],
						int64(sp.K), int64(sp.K*2), int64(sp.N*4))
				}
			}
		}(wid)
	}
	wg.Wait()
}

func gemmF16OuterTile(A, Bp []uint16, C []float32, M, N, K, nthreads, tileN int, kernel func(a, bp *uint16, c *float32, K, lda, ldc int64)) {
	mblocks := M / 4
	fullNTiles := N / tileN
	tailN := N - fullNTiles*tileN
	work := func(mb0, mb1 int) {
		var tailTile [4 * 32]float32
		for nb0 := 0; nb0 < fullNTiles; nb0 += outerCacheBlockTiles {
			nb1 := minInt(fullNTiles, nb0+outerCacheBlockTiles)
			for mb := mb0; mb < mb1; mb++ {
				m := mb * 4
				for nt := nb0; nt < nb1; nt++ {
					kernel(&A[m*K], &Bp[nt*K*tileN], &C[m*N+nt*tileN],
						int64(K), int64(K*2), int64(N*4))
				}
			}
		}
		if tailN == 0 {
			return
		}
		for mb := mb0; mb < mb1; mb++ {
			m := mb * 4
			kernel(&A[m*K], &Bp[fullNTiles*K*tileN], &tailTile[0],
				int64(K), int64(K*2), int64(tileN*4))
			copyF32TailTile(C[m*N+fullNTiles*tileN:], N, tailN, tileN, tailTile[:])
		}
	}
	if nthreads <= 1 {
		work(0, mblocks)
	} else {
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
	if mblocks*4 < M {
		gemmF16PackedRowsScalar(A, Bp, C, mblocks*4, M, N, K, tileN)
	}
}

// GemmF16Threaded runs GemmF16 across nthreads goroutines partitioned over M
// rows. It dispatches to the tiled M4xN32 kernel when there is useful tile work,
// falls back to M4xN16 for medium-width cases, and finally to the dot-loop
// kernel for very small shapes.
func GemmF16Threaded(A, B []uint16, C []float32, M, N, K, nthreads int) {
	if M >= 4 && N >= 32 {
		GemmF16Outer32(A, PackBF16N32(B, N, K), C, M, N, K, nthreads)
		return
	}
	if M >= 4 && N >= 16 {
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
