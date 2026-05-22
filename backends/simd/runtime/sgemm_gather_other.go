//go:build !amd64 && !riscv64

package simd

import "unsafe"

// gatherMicroKernel6x8 is the non-amd64 fallback for SgemmNTGather.
// arm64 has HasSgemmAsm=true, so this must be correct rather than a panic
// even though it is not SIMD-accelerated yet.
func gatherMicroKernel6x8(k int, alpha float32, a unsafe.Pointer, lda int, b unsafe.Pointer, indices unsafe.Pointer, c unsafe.Pointer, ldc int) {
	if k <= 0 || a == nil || b == nil || indices == nil || c == nil || lda < k || ldc < 8 {
		return
	}
	idx := unsafe.Slice((*int32)(indices), 8)
	for i := 0; i < 6; i++ {
		for j := 0; j < 8; j++ {
			sum := float32(0)
			for p := 0; p < k; p++ {
				aOff, okA := checkedFloat32ByteOffset(i*lda + p)
				bOff, okB := checkedFloat32ByteOffset(int(idx[j]) + p)
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
