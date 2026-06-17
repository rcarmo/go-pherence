package nvidia

import (
	"fmt"
	"unsafe"
)

func VecAddF32Buffer(a, b, out *Buffer, n int) error {
	if n <= 0 {
		return nil
	}
	if fnVecAdd == 0 || !SgemmReady() || a == nil || b == nil || out == nil || a.Ptr == 0 || b.Ptr == 0 || out.Ptr == 0 || a.Size < n*4 || b.Size < n*4 || out.Size < n*4 {
		return fmt.Errorf("invalid F32 vec-add buffers")
	}
	nn := uint32(n)
	grid := uint32((n + 255) / 256)
	return LaunchKernel(fnVecAdd, grid, 1, 1, 256, 1, 1, 0, unsafe.Pointer(&a.Ptr), unsafe.Pointer(&b.Ptr), unsafe.Pointer(&out.Ptr), unsafe.Pointer(&nn))
}
