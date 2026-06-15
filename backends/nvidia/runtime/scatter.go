package nvidia

import (
	"fmt"
	"unsafe"
)

var (
	fnScatterWeightedRows      CUfunction
	fnScatterWeightedRowsBatch CUfunction
)

func ScatterWeightedRowsBatch(dst, src, pos, weights *Buffer, rows, hidden int) error {
	if rows <= 0 || hidden <= 0 {
		return nil
	}
	need := rows * hidden
	if dst == nil || src == nil || pos == nil || weights == nil || dst.Ptr == 0 || src.Ptr == 0 || pos.Ptr == 0 || weights.Ptr == 0 || src.Size < need*4 || pos.Size < rows*4 || weights.Size < rows*4 {
		return fmt.Errorf("invalid scatter weighted rows batch buffers rows=%d hidden=%d", rows, hidden)
	}
	if fnScatterWeightedRowsBatch == 0 {
		return fmt.Errorf("scatter weighted rows batch kernel not loaded")
	}
	r := uint32(rows)
	h := uint32(hidden)
	grid := uint32((need + 255) / 256)
	return LaunchKernel(fnScatterWeightedRowsBatch, grid, 1, 1, 256, 1, 1, 0, unsafe.Pointer(&dst.Ptr), unsafe.Pointer(&src.Ptr), unsafe.Pointer(&pos.Ptr), unsafe.Pointer(&weights.Ptr), unsafe.Pointer(&r), unsafe.Pointer(&h))
}

func ScatterWeightedRows(dst, src, pos *Buffer, weight float32, rows, hidden int) error {
	if rows <= 0 || hidden <= 0 {
		return nil
	}
	need := rows * hidden
	if dst == nil || src == nil || pos == nil || dst.Ptr == 0 || src.Ptr == 0 || pos.Ptr == 0 || src.Size < need*4 || pos.Size < rows*4 {
		return fmt.Errorf("invalid scatter weighted rows buffers rows=%d hidden=%d", rows, hidden)
	}
	if fnScatterWeightedRows == 0 {
		return fmt.Errorf("scatter weighted rows kernel not loaded")
	}
	r := uint32(rows)
	h := uint32(hidden)
	grid := uint32((need + 255) / 256)
	return LaunchKernel(fnScatterWeightedRows, grid, 1, 1, 256, 1, 1, 0, unsafe.Pointer(&dst.Ptr), unsafe.Pointer(&src.Ptr), unsafe.Pointer(&pos.Ptr), unsafe.Pointer(&weight), unsafe.Pointer(&r), unsafe.Pointer(&h))
}
