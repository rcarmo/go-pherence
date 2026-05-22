//go:build riscv64

package simd

import "unsafe"

func gatherMicroKernel6x8(k int, alpha float32, a unsafe.Pointer, lda int, b unsafe.Pointer, indices unsafe.Pointer, c unsafe.Pointer, ldc int) {
	if k <= 0 || a == nil || b == nil || indices == nil || c == nil || lda < k || ldc < 8 {
		return
	}
	idx := unsafe.Slice((*int32)(indices), 8)
	for i := 0; i < 6; i++ {
		for j := 0; j < 8; j++ {
			sum := float32(0)
			for p := 0; p < k; p++ {
				sum += loadF32(a, i*lda+p) * loadF32(b, int(idx[j])+p)
			}
			storeF32(c, i*ldc+j, loadF32(c, i*ldc+j)+alpha*sum)
		}
	}
}
