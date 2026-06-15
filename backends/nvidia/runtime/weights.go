package nvidia

import (
	"fmt"
	"unsafe"
)

var fnMulWeights CUfunction

func MulWeights(out, a, b *Buffer, n int) error {
	if n <= 0 {
		return nil
	}
	if out == nil || a == nil || b == nil || out.Ptr == 0 || a.Ptr == 0 || b.Ptr == 0 || out.Size < n*4 || a.Size < n*4 || b.Size < n*4 {
		return fmt.Errorf("invalid mul weights buffers n=%d", n)
	}
	if fnMulWeights == 0 {
		return fmt.Errorf("mul weights kernel not loaded")
	}
	nn := uint32(n)
	grid := uint32((n + 255) / 256)
	return LaunchKernel(fnMulWeights, grid, 1, 1, 256, 1, 1, 0, unsafe.Pointer(&out.Ptr), unsafe.Pointer(&a.Ptr), unsafe.Pointer(&b.Ptr), unsafe.Pointer(&nn))
}
