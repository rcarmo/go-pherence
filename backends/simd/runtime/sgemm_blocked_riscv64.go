//go:build riscv64

package simd

import "unsafe"

func sgemmNTTileFMA(iLen, jLen, kLen int, alpha float32, a unsafe.Pointer, lda int, b unsafe.Pointer, ldb int, c unsafe.Pointer, ldc int) {
	if iLen <= 0 || jLen <= 0 || kLen <= 0 || a == nil || b == nil || c == nil || lda < kLen || ldb < kLen || ldc < jLen {
		return
	}
	for i := 0; i < iLen; i++ {
		for j := 0; j < jLen; j++ {
			sum := float32(0)
			for p := 0; p < kLen; p++ {
				sum += loadF32(a, i*lda+p) * loadF32(b, j*ldb+p)
			}
			storeF32(c, i*ldc+j, loadF32(c, i*ldc+j)+alpha*sum)
		}
	}
}
