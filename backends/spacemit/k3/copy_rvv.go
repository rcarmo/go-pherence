package k3

import "unsafe"

//go:noescape
func copyBytesRVV(dst, src *byte, n int)

func copyTCMBytes(dst, src []byte) {
	if len(src) == 0 {
		return
	}
	if len(dst) < len(src) {
		panic("copyTCMBytes: short dst")
	}
	// TCM staging copies in the hot wave path are naturally 128-byte aligned
	// sizes (Q4_K N32 blocks and INT8 8-row tiles). Use RVV for those and
	// preserve normal copy() as a safe fallback for odd sizes.
	if len(src)%128 == 0 {
		copyBytesRVV((*byte)(unsafe.Pointer(&dst[0])), (*byte)(unsafe.Pointer(&src[0])), len(src))
		return
	}
	copy(dst, src)
}
