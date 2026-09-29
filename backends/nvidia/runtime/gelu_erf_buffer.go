package nvidia

import (
	"fmt"
	"unsafe"
)

// GELUErfF32Buffer applies the exact-form erf GELU approximation in place to
// n GPU-resident F32 values. It returns an error rather than falling back to CPU.
func GELUErfF32Buffer(x *Buffer, n int) error {
	if n <= 0 || !fitsUint32(n) || !whisperF32Extent(x, n) {
		return fmt.Errorf("invalid F32 erf-GELU device buffer")
	}
	if !SgemmReady() || fnGELUErf == 0 {
		return fmt.Errorf("F32 erf-GELU kernel unavailable")
	}
	grid, ok := grid1DFor(n, 256)
	if !ok {
		return fmt.Errorf("F32 erf-GELU grid overflow")
	}
	nn := uint32(n)
	return LaunchKernel(fnGELUErf, grid, 1, 1, 256, 1, 1, 0,
		unsafe.Pointer(&x.Ptr), unsafe.Pointer(&nn))
}
