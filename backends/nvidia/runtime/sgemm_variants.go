package nvidia

import (
	"fmt"
	"unsafe"
)

var (
	sgemmOracleFn, sgemmReg2Fn, sgemmSkinnyFn CUfunction
)

func launchSgemmVariant(fn CUfunction, variant string, M, N, K int, alpha float32, A, B, C *Buffer) error {
	if fn == 0 {
		return fmt.Errorf("SGEMM %s unavailable", variant)
	}
	m, n, k := uint32(M), uint32(N), uint32(K)
	args := []unsafe.Pointer{unsafe.Pointer(&A.Ptr), unsafe.Pointer(&B.Ptr), unsafe.Pointer(&C.Ptr), unsafe.Pointer(&m), unsafe.Pointer(&n), unsafe.Pointer(&k), unsafe.Pointer(&alpha)}
	var gx, gy, tx, ty uint32
	switch variant {
	case "reg2":
		gx = uint32((N + 15) / 16)
		gy = uint32((M + 15) / 16)
		tx = 8
		ty = 16
	case "skinny":
		gx = uint32((N + 31) / 32)
		gy = uint32((M + 3) / 4)
		tx = 32
		ty = 4
	default:
		gx = uint32((N + 15) / 16)
		gy = uint32((M + 15) / 16)
		tx = 16
		ty = 16
	}
	return LaunchKernel(fn, gx, gy, 1, tx, ty, 1, 0, args...)
}
func sgemmVariant(M, N, K int) (CUfunction, string) {
	// Candidate variants remain loaded for A/B benchmarking, but live parity
	// must pass every shape before either can become default.
	return sgemmOracleFn, "oracle"
}
