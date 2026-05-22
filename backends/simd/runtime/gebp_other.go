//go:build !amd64 && !arm64 && !riscv64

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
				aOff, okA := checkedFloat32ByteOffset(i*lda + p)
				bOff, okB := checkedFloat32ByteOffset(p*gebpNR + j)
				if !okA || !okB {
					return
				}
				sum += *(*float32)(unsafe.Add(a, aOff)) * *(*float32)(unsafe.Add(bp, bOff))
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
