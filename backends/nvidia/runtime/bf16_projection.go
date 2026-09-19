package nvidia

import (
	"fmt"
	"math"
	"unsafe"

	"github.com/rcarmo/go-pherence/internal/checked"
)

var fnWidenBF16Transpose CUfunction

// WidenBF16Transpose expands compact row-major BF16 weights [rows,cols] into
// caller-owned F32 [cols,rows] scratch for the existing checked SGEMM path.
// No host copies, vocabulary projection, silent fallback or persistent F32
// weight allocation occurs. Caller serialises scratch use and owns its buffers.
func WidenBF16Transpose(dst, src *Buffer, rows, cols int) error {
	n, ok := checked.MulInt(rows, cols)
	if !ok || rows <= 0 || cols <= 0 || uint64(n) > math.MaxUint32 {
		return fmt.Errorf("invalid BF16 projection shape")
	}
	bytes, ok := checked.MulInt(n, 4)
	if !ok || src == nil || dst == nil || src.Ptr == 0 || dst.Ptr == 0 || src.Size < n*2 || dst.Size < bytes {
		return fmt.Errorf("invalid BF16 projection buffers")
	}
	if dst.Ptr == src.Ptr {
		return fmt.Errorf("BF16 widening cannot alias input")
	}
	if !SgemmReady() || fnWidenBF16Transpose == 0 {
		return fmt.Errorf("CUDA BF16 widening unavailable")
	}
	r, c := uint32(rows), uint32(cols)
	return LaunchKernel(fnWidenBF16Transpose, uint32((uint64(n)+255)/256), 1, 1, 256, 1, 1, 0, unsafe.Pointer(&src.Ptr), unsafe.Pointer(&dst.Ptr), unsafe.Pointer(&r), unsafe.Pointer(&c))
}
