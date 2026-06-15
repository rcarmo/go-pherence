package nvidia

import (
	"fmt"
	"unsafe"
)

var fnGateUpGELU CUfunction

func GateUpGELUBuffer(src, out *Buffer, batch, intermediate int) error {
	if batch <= 0 || intermediate <= 0 {
		return nil
	}
	need := batch * intermediate
	if src == nil || out == nil || src.Ptr == 0 || out.Ptr == 0 || src.Size < need*2*4 || out.Size < need*4 {
		return fmt.Errorf("invalid gate/up GELU buffers batch=%d intermediate=%d", batch, intermediate)
	}
	if fnGateUpGELU == 0 {
		return fmt.Errorf("gate/up GELU kernel not loaded")
	}
	b := uint32(batch)
	i := uint32(intermediate)
	grid := uint32((need + 255) / 256)
	return LaunchKernel(fnGateUpGELU, grid, 1, 1, 256, 1, 1, 0, unsafe.Pointer(&src.Ptr), unsafe.Pointer(&out.Ptr), unsafe.Pointer(&b), unsafe.Pointer(&i))
}
