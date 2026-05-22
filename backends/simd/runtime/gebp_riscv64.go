//go:build riscv64

package simd

import "unsafe"

const gebpMR = 4

func gebpMicroKernel(k int, alpha float32, a unsafe.Pointer, lda int, bp unsafe.Pointer, c unsafe.Pointer, ldc int) {
	if k <= 0 || a == nil || bp == nil || c == nil || lda < k || ldc < gebpNR {
		return
	}
	for i := 0; i < gebpMR; i++ {
		for j := 0; j < gebpNR; j++ {
			sum := float32(0)
			for p := 0; p < k; p++ {
				sum += loadF32(a, i*lda+p) * loadF32(bp, p*gebpNR+j)
			}
			storeF32(c, i*ldc+j, loadF32(c, i*ldc+j)+alpha*sum)
		}
	}
}
