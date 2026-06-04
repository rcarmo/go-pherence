//go:build riscv64

package simd

import "unsafe"

func sgemmNTTileFMA(iLen, jLen, kLen int, alpha float32, a unsafe.Pointer, lda int, b unsafe.Pointer, ldb int, c unsafe.Pointer, ldc int) {
	if iLen <= 0 || jLen <= 0 || kLen <= 0 || a == nil || b == nil || c == nil || lda < kLen || ldb < kLen || ldc < jLen {
		return
	}
	for i := 0; i < iLen; i++ {
		aOff, okA := checkedFloat32ByteOffset(i * lda)
		if !okA {
			return
		}
		aRow := unsafe.Slice((*float32)(unsafe.Add(a, aOff)), kLen)
		for j := 0; j < jLen; j++ {
			bOff, okB := checkedFloat32ByteOffset(j * ldb)
			if !okB {
				return
			}
			bRow := unsafe.Slice((*float32)(unsafe.Add(b, bOff)), kLen)
			sum := sdotM4Asm(aRow, bRow)
			storeF32(c, i*ldc+j, loadF32(c, i*ldc+j)+alpha*sum)
		}
	}
}
