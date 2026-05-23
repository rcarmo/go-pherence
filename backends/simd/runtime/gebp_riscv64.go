//go:build riscv64

package simd

import "unsafe"

const gebpMR = 4

func gebpMicroKernel(k int, alpha float32, a unsafe.Pointer, lda int, bp unsafe.Pointer, c unsafe.Pointer, ldc int) {
	if k <= 0 || a == nil || bp == nil || c == nil || lda < k || ldc < gebpNR {
		return
	}
	bCol := make([]float32, k)
	for j := 0; j < gebpNR; j++ {
		for p := 0; p < k; p++ {
			bCol[p] = loadF32(bp, p*gebpNR+j)
		}
		for i := 0; i < gebpMR; i++ {
			aOff, okA := checkedFloat32ByteOffset(i * lda)
			if !okA {
				return
			}
			aRow := unsafe.Slice((*float32)(unsafe.Add(a, aOff)), k)
			sum := sdotAsm(aRow, bCol)
			storeF32(c, i*ldc+j, loadF32(c, i*ldc+j)+alpha*sum)
		}
	}
}
