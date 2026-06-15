package nvidia

import (
	"fmt"
	"unsafe"
)

var fnSplitGateUp CUfunction

func SplitGateUpBuffer(src, gate, up *Buffer, batch, intermediate int) error {
	if batch <= 0 || intermediate <= 0 {
		return nil
	}
	need := batch * intermediate
	if src == nil || gate == nil || up == nil || src.Ptr == 0 || gate.Ptr == 0 || up.Ptr == 0 || src.Size < need*2*4 || gate.Size < need*4 || up.Size < need*4 {
		return fmt.Errorf("invalid split gate/up buffers batch=%d intermediate=%d", batch, intermediate)
	}
	if fnSplitGateUp == 0 {
		return fmt.Errorf("split gate/up kernel not loaded")
	}
	b := uint32(batch)
	i := uint32(intermediate)
	grid := uint32((need + 255) / 256)
	return LaunchKernel(fnSplitGateUp, grid, 1, 1, 256, 1, 1, 0, unsafe.Pointer(&src.Ptr), unsafe.Pointer(&gate.Ptr), unsafe.Pointer(&up.Ptr), unsafe.Pointer(&b), unsafe.Pointer(&i))
}
