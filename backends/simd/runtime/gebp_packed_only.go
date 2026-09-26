package simd

import (
	"unsafe"

	"github.com/rcarmo/go-pherence/internal/checked"
)

// SgemmNTPackedOnlyTo computes C += alpha*A*B^T for row-major A[m,lda] and
// C[m,ldc], consuming only prepacked full 16-column NT weight panels in
// packed[panel][k][0:16] order as produced by PackSgemmNTWeights.
//
// Unlike SgemmNTPrepackedTo, this entrypoint accepts only complete panels:
// n must be positive and divisible by 16. packed is treated as read-only and
// may alias A, but C must be disjoint from both inputs. Invalid inputs return
// false before any writes. The scalar path and amd64/ARM64 microkernels
// allocate nothing; the existing RISC-V microkernel allocates a column buffer.
func SgemmNTPackedOnlyTo(c, a, packed []float32, m, n, k int, alpha float32, lda, ldc int) bool {
	fullPanels, packedLen, ok := validSgemmNTPackedOnlyArgs(c, a, packed, m, n, k, lda, ldc)
	if !ok {
		return false
	}
	pp := packed[:packedLen]
	if !float32SlicesDisjoint(c, a) || !float32SlicesDisjoint(c, pp) {
		return false
	}
	panelStride := packedLen / fullPanels
	if !HasSgemmAsm {
		for panel := 0; panel < fullPanels; panel++ {
			jj := panel * gebpNR
			bp := pp[panel*panelStride : (panel+1)*panelStride]
			sgemmNTPackedOnlyPanelScalar(c[jj:], a, bp, m, k, alpha, lda, ldc)
		}
		return true
	}
	for panel := 0; panel < fullPanels; panel++ {
		jj := panel * gebpNR
		bp := pp[panel*panelStride : (panel+1)*panelStride]
		ii := 0
		for ; ii+gebpMR <= m; ii += gebpMR {
			gebpMicroKernel(k, alpha, unsafe.Pointer(&a[ii*lda]), lda, unsafe.Pointer(&bp[0]), unsafe.Pointer(&c[ii*ldc+jj]), ldc)
		}
		if ii < m {
			sgemmNTPackedOnlyPanelScalar(c[ii*ldc+jj:], a[ii*lda:], bp, m-ii, k, alpha, lda, ldc)
		}
	}
	return true
}

func sgemmNTPackedOnlyPanelScalar(c, a, bp []float32, m, k int, alpha float32, lda, ldc int) {
	for i := 0; i < m; i++ {
		aRow := a[i*lda:]
		cRow := c[i*ldc:]
		for d := 0; d < gebpNR; d++ {
			sum := float32(0)
			for p := 0; p < k; p++ {
				sum += aRow[p] * bp[p*gebpNR+d]
			}
			cRow[d] += alpha * sum
		}
	}
}

func validSgemmNTPackedOnlyArgs(c, a, packed []float32, m, n, k, lda, ldc int) (fullPanels, packedLen int, ok bool) {
	if m <= 0 || n <= 0 || k <= 0 || n%gebpNR != 0 || lda < k || ldc < n {
		return 0, 0, false
	}
	aBase, okABase := checked.MulInt(m-1, lda)
	aNeed, okA := checked.AddInt(aBase, k)
	cBase, okCBase := checked.MulInt(m-1, ldc)
	cNeed, okC := checked.AddInt(cBase, n)
	packedNeed, okPackedNeed := checked.MulInt(n, k)
	fullPanels, packedLen, okLayout := checkedSgemmNTFullPanelLayout(n, k)
	if !okABase || !okA || !okCBase || !okC || !okPackedNeed || !okLayout || fullPanels == 0 || packedNeed != packedLen {
		return 0, 0, false
	}
	if len(a) < aNeed || len(c) < cNeed || len(packed) < packedNeed {
		return 0, 0, false
	}
	return fullPanels, packedLen, true
}
