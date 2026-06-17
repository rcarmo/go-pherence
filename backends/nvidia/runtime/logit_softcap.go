package nvidia

import (
	"fmt"
	"unsafe"
)

var fnLogitSoftcapF32 CUfunction

func LogitSoftcapF32(buf *Buffer, n int, cap float32) error {
	if n <= 0 || cap <= 0 {
		return nil
	}
	if buf == nil || buf.Ptr == 0 || buf.Size < n*4 || fnLogitSoftcapF32 == 0 || !SgemmReady() {
		return fmt.Errorf("invalid logit softcap buffer")
	}
	nn := uint32(n)
	cc := cap
	args := []unsafe.Pointer{unsafe.Pointer(&buf.Ptr), unsafe.Pointer(&nn), unsafe.Pointer(&cc)}
	grid := uint32((n + 255) / 256)
	return LaunchKernel(fnLogitSoftcapF32, grid, 1, 1, 256, 1, 1, 0, args...)
}
