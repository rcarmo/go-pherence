package simd

import (
	"fmt"

	"github.com/rcarmo/go-pherence/internal/checked"
	"unsafe"
)

// PackSgemmNTWeights packs the full 16-column panels of row-major NT weights
// B[n,ldb] into the panel-major layout consumed by SgemmNTPrepackedTo:
// packed[panel][k][0:16]. Only complete 16-column panels are included, so the
// returned slice length is floor(n/16)*k*16. The source weights are still
// required at multiply time for any N tail and for checked scalar fallback.
//
// Caller contract:
//   - n, k, and ldb describe the raw NT weights as in SgemmNTTo
//   - ldb must be at least k
//   - the returned buffer is read-only input to SgemmNTPrepackedTo
func PackSgemmNTWeights(weights []float32, n, k, ldb int) ([]float32, error) {
	fullPanels, packedLen, err := validatePackSgemmNTWeights(weights, n, k, ldb)
	if err != nil {
		return nil, err
	}
	packed := make([]float32, packedLen)
	if _, err := packSgemmNTWeightsIntoChecked(weights, k, ldb, packed, fullPanels, packedLen); err != nil {
		return nil, err
	}
	return packed, nil
}

// PackSgemmNTWeightsInto is the allocation-free form of PackSgemmNTWeights.
// dst must have room for floor(n/16)*k*16 float32 values and must not overlap
// the source weights. The returned slice is dst[:floor(n/16)*k*16] for call
// chaining. When n < 16 the returned slice is empty and dst is untouched.
func PackSgemmNTWeightsInto(weights []float32, n, k, ldb int, dst []float32) ([]float32, error) {
	fullPanels, packedLen, err := validatePackSgemmNTWeights(weights, n, k, ldb)
	if err != nil {
		return nil, err
	}
	return packSgemmNTWeightsIntoChecked(weights, k, ldb, dst, fullPanels, packedLen)
}

// SgemmNTPrepackedTo computes C += alpha*A*B^T for row-major A[m,lda],
// weights B[n,ldb], and C[m,ldc], reusing prepacked full 16-column B panels
// produced by PackSgemmNTWeights or PackSgemmNTWeightsInto.
//
// packed stores only the complete floor(n/16) panels in panel-major
// packed[panel][k][0:16] order. Any remaining N tail columns and any M tail
// rows are computed from the raw weights through the checked fallback path so
// the full source weights slice is always required. The function validates all
// raw and packed slice footprints before any SIMD or scalar work and allocates
// nothing on success.
//
// Caller contract:
//   - c, a, and weights must satisfy the same shape rules as SgemmNTTo
//   - packed must contain at least floor(n/16)*k*16 values from the pack API
//   - packed is treated as read-only and must not alias c or a
func SgemmNTPrepackedTo(c, a, weights, packed []float32, m, n, k int, alpha float32, lda, ldb, ldc int) bool {
	if !validSgemmSliceArgs(c, a, weights, m, n, k, lda, ldb, ldc, true) {
		return false
	}
	fullPanels, packedLen, ok := checkedSgemmNTFullPanelLayout(n, k)
	if !ok || len(packed) < packedLen {
		return false
	}
	if packedLen > 0 && (!float32SlicesDisjoint(packed[:packedLen], c) || !float32SlicesDisjoint(packed[:packedLen], a)) {
		return false
	}
	if !HasSgemmAsm || m < gebpMR || fullPanels == 0 {
		return SgemmNTTo(c, a, weights, m, n, k, alpha, lda, ldb, ldc)
	}

	panelStride := k * gebpNR
	for panel := 0; panel < fullPanels; panel++ {
		jj := panel * gebpNR
		bp := packed[panel*panelStride : (panel+1)*panelStride]
		ii := 0
		for ; ii+gebpMR <= m; ii += gebpMR {
			gebpMicroKernel(k, alpha, unsafe.Pointer(&a[ii*lda]), lda, unsafe.Pointer(&bp[0]), unsafe.Pointer(&c[ii*ldc+jj]), ldc)
		}
		if ii < m {
			if !SgemmNTTo(c[ii*ldc+jj:], a[ii*lda:], weights[jj*ldb:], m-ii, gebpNR, k, alpha, lda, ldb, ldc) {
				return false
			}
		}
	}
	if tailCols := n - fullPanels*gebpNR; tailCols > 0 {
		return SgemmNTTo(c[fullPanels*gebpNR:], a, weights[fullPanels*gebpNR*ldb:], m, tailCols, k, alpha, lda, ldb, ldc)
	}
	return true
}

func packSgemmNTWeightsIntoChecked(weights []float32, k, ldb int, dst []float32, fullPanels, packedLen int) ([]float32, error) {
	if len(dst) < packedLen {
		return nil, fmt.Errorf("simd: packed NT weights buffer too short: have %d need %d", len(dst), packedLen)
	}
	out := dst[:packedLen]
	if packedLen == 0 {
		return out, nil
	}
	if !float32SlicesDisjoint(out, weights) {
		return nil, fmt.Errorf("simd: packed NT weights buffer must not overlap source weights")
	}
	panelStride := k * gebpNR
	for panel := 0; panel < fullPanels; panel++ {
		packBNT(weights, ldb, panel*gebpNR, gebpNR, k, out[panel*panelStride:(panel+1)*panelStride])
	}
	return out, nil
}

func validatePackSgemmNTWeights(weights []float32, n, k, ldb int) (fullPanels, packedLen int, err error) {
	if n <= 0 || k <= 0 {
		return 0, 0, fmt.Errorf("simd: invalid NT weight shape n=%d k=%d", n, k)
	}
	if ldb < k {
		return 0, 0, fmt.Errorf("simd: invalid NT weights stride ldb=%d for k=%d", ldb, k)
	}
	weightBase, okBase := checked.MulInt(n-1, ldb)
	weightNeed, okNeed := checked.AddInt(weightBase, k)
	if !okBase || !okNeed {
		return 0, 0, fmt.Errorf("simd: NT weights footprint overflows int for n=%d k=%d ldb=%d", n, k, ldb)
	}
	fullPanels, packedLen, okPacked := checkedSgemmNTFullPanelLayout(n, k)
	if !okPacked {
		return 0, 0, fmt.Errorf("simd: packed NT weights footprint overflows int for n=%d k=%d", n, k)
	}
	if len(weights) < weightNeed {
		return 0, 0, fmt.Errorf("simd: NT weights slice too short: have %d need %d", len(weights), weightNeed)
	}
	return fullPanels, packedLen, nil
}

func checkedSgemmNTFullPanelLayout(n, k int) (fullPanels, packedLen int, ok bool) {
	if n <= 0 || k <= 0 {
		return 0, 0, false
	}
	fullPanels = n / gebpNR
	panelStride, okStride := checked.MulInt(k, gebpNR)
	if !okStride {
		return 0, 0, false
	}
	packedLen, ok = checked.MulInt(fullPanels, panelStride)
	return fullPanels, packedLen, ok
}
