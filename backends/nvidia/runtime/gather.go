package nvidia

import (
	"fmt"
	"unsafe"
)

var fnGatherRows CUfunction

func GatherRows(dst, src, pos *Buffer, rows, hidden int) error {
	if rows <= 0 || hidden <= 0 {
		return nil
	}
	need := rows * hidden
	if dst == nil || src == nil || pos == nil || dst.Ptr == 0 || src.Ptr == 0 || pos.Ptr == 0 || dst.Size < need*4 || pos.Size < rows*4 {
		return fmt.Errorf("invalid gather rows buffers rows=%d hidden=%d", rows, hidden)
	}
	if fnGatherRows == 0 {
		return fmt.Errorf("gather rows kernel not loaded")
	}
	r := uint32(rows)
	h := uint32(hidden)
	grid := uint32((need + 255) / 256)
	return LaunchKernel(fnGatherRows, grid, 1, 1, 256, 1, 1, 0, unsafe.Pointer(&dst.Ptr), unsafe.Pointer(&src.Ptr), unsafe.Pointer(&pos.Ptr), unsafe.Pointer(&r), unsafe.Pointer(&h))
}
