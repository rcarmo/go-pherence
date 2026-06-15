package nvidia

import (
	"fmt"
	"unsafe"
)

// GELUTanhMulBuffer computes gate = GELU(gate) * up in-place on device buffers.
func GELUTanhMulBuffer(gate, up *Buffer, n int) error {
	if n < 0 || gate == nil || up == nil || gate.Ptr == 0 || up.Ptr == 0 || gate.Size < n*4 || up.Size < n*4 {
		return fmt.Errorf("invalid GELU buffer inputs n=%d", n)
	}
	if n == 0 {
		return nil
	}
	if fnGELUTanhMul == 0 {
		return fmt.Errorf("GELU tanh mul kernel not loaded")
	}
	nn := uint32(n)
	grid := uint32((n + 255) / 256)
	return LaunchKernel(fnGELUTanhMul, grid, 1, 1, 256, 1, 1, 0, unsafe.Pointer(&gate.Ptr), unsafe.Pointer(&up.Ptr), unsafe.Pointer(&nn))
}
