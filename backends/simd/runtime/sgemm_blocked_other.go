//go:build !amd64 && !arm64 && !riscv64

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
				aOff, okA := checkedFloat32ByteOffset(i*lda + p)
				bOff, okB := checkedFloat32ByteOffset(j*ldb + p)
				if !okA || !okB {
					return
				}
				sum += *(*float32)(unsafe.Add(a, aOff)) * *(*float32)(unsafe.Add(b, bOff))
			}
			cOff, okC := checkedFloat32ByteOffset(i*ldc + j)
			if !okC {
				return
			}
			cv := (*float32)(unsafe.Add(c, cOff))
			*cv += alpha * sum
		}
	}
}
