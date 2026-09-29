package nvidia

import (
	"fmt"
	"math"
	"unsafe"

	"github.com/rcarmo/go-pherence/internal/checked"
)

var fnAffineLayerNormF32 CUfunction

func deviceRangesOverlap(a CUdeviceptr, aBytes uint64, b CUdeviceptr, bBytes uint64) bool {
	aStart, bStart := uint64(a), uint64(b)
	// Reject wraparound rather than allowing an invalid address range.
	if aStart+aBytes < aStart || bStart+bBytes < bStart {
		return true
	}
	return aStart < bStart+bBytes && bStart < aStart+aBytes
}

// AffineLayerNormF32Buffer normalizes row-major GPU-resident F32 input and
// applies per-column gamma and beta. Output may alias input, but not the
// gamma/beta buffers; non-aliased inputs remain unchanged.
func AffineLayerNormF32Buffer(out, x, gamma, beta *Buffer, rows, cols int, eps float32) error {
	if rows <= 0 || rows > 65535 || cols <= 0 || !fitsUint32(cols) || !(eps > 0) || math.IsInf(float64(eps), 0) {
		return fmt.Errorf("invalid affine LayerNorm dimensions or epsilon")
	}
	n, ok := checked.MulInt(rows, cols)
	if !ok || !fitsUint32(n) || !whisperF32Extent(out, rows, cols) || !whisperF32Extent(x, rows, cols) ||
		!whisperF32Extent(gamma, cols) || !whisperF32Extent(beta, cols) {
		return fmt.Errorf("invalid affine LayerNorm device buffers")
	}
	outBytes, affineBytes := uint64(n)*4, uint64(cols)*4
	if deviceRangesOverlap(out.Ptr, outBytes, gamma.Ptr, affineBytes) ||
		deviceRangesOverlap(out.Ptr, outBytes, beta.Ptr, affineBytes) ||
		(out.Ptr != x.Ptr && deviceRangesOverlap(out.Ptr, outBytes, x.Ptr, outBytes)) {
		return fmt.Errorf("overlapping affine LayerNorm device buffers")
	}
	if !SgemmReady() || fnAffineLayerNormF32 == 0 {
		return fmt.Errorf("affine LayerNorm kernel unavailable")
	}
	rr, cc := uint32(rows), uint32(cols)
	return LaunchKernel(fnAffineLayerNormF32, rr, 1, 1, 256, 1, 1, 0,
		unsafe.Pointer(&out.Ptr), unsafe.Pointer(&x.Ptr), unsafe.Pointer(&gamma.Ptr), unsafe.Pointer(&beta.Ptr),
		unsafe.Pointer(&rr), unsafe.Pointer(&cc), unsafe.Pointer(&eps))
}
