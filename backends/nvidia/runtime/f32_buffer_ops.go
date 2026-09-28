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

func VecScaleF32Buffer(src, out *Buffer, n int, scale float32) error {
	if n <= 0 {
		return nil
	}
	if fnVecScale == 0 || !SgemmReady() || src == nil || out == nil || src.Ptr == 0 || out.Ptr == 0 || src.Size < n*4 || out.Size < n*4 {
		return fmt.Errorf("invalid F32 vec-scale buffers")
	}
	nn := uint32(n)
	grid := uint32((n + 255) / 256)
	return LaunchKernel(fnVecScale, grid, 1, 1, 256, 1, 1, 0, unsafe.Pointer(&src.Ptr), unsafe.Pointer(&out.Ptr), unsafe.Pointer(&scale), unsafe.Pointer(&nn))
}
