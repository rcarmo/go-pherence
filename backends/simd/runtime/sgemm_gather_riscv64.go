//go:build riscv64

package simd

import "unsafe"

func gatherMicroKernel6x8(k int, alpha float32, a unsafe.Pointer, lda int, b unsafe.Pointer, indices unsafe.Pointer, c unsafe.Pointer, ldc int) {
	if k <= 0 || a == nil || b == nil || indices == nil || c == nil || lda < k || ldc < 8 {
		return
	}
	idx := unsafe.Slice((*int32)(indices), 8)
	for i := 0; i < 6; i++ {
		aOff, okA := checkedFloat32ByteOffset(i * lda)
		if !okA {
			return
		}
		aRow := unsafe.Slice((*float32)(unsafe.Add(a, aOff)), k)
		for j := 0; j < 8; j++ {
			bOff, okB := checkedFloat32ByteOffset(int(idx[j]))
			if !okB {
				return
			}
			bRow := unsafe.Slice((*float32)(unsafe.Add(b, bOff)), k)
			sum := sdotAsm(aRow, bRow)
			storeF32(c, i*ldc+j, loadF32(c, i*ldc+j)+alpha*sum)
		}
	}
}
