//go:build riscv64

// Package ime2 provides pure Go access to SpacemiT IME2 matrix instructions.
// These are RISC-V custom vector instructions (vmadot family) that perform
// INT8 matrix multiply-accumulate operations using on-chip hardware.
//
// Requires X100 cores (processors 0-7 on SpacemiT K3).
package ime2

import "unsafe"

// vmadotSS4x8 computes C[4x4] += A[4x8] * B[4x8]^T
// using signed*signed int8 multiply with int32 accumulation.
// A and B must be 32 bytes each. C must be 64 bytes (16 int32s).
//
//go:noescape
func vmadotSS4x8(A *byte, B *byte, C *int32)

// vmadotUS4x8 computes C[4x4] += A[4x8] * B[4x8]^T
// using unsigned*signed int8 multiply with int32 accumulation.
//
//go:noescape
func vmadotUS4x8(A *byte, B *byte, C *int32)

// vmadotAccSS4x8 loads existing C, accumulates A*B^T, stores back.
// This is the version used in loop bodies where C accumulates across K tiles.
//
//go:noescape
func vmadotAccSS4x8(A *byte, B *byte, C *int32)

// Matmul4x8 performs a single 4x8 x 8x4 matrix multiply (signed x signed).
// Result is 4x4 int32 matrix.
func Matmul4x8(A, B []int8, C []int32) {
	if len(A) < 32 || len(B) < 32 || len(C) < 16 {
		panic("ime2: buffer too small")
	}
	vmadotSS4x8((*byte)(unsafe.Pointer(&A[0])), (*byte)(unsafe.Pointer(&B[0])), &C[0])
}

// VmadotAccSS4x8 is the exported version of vmadotAccSS4x8.
func VmadotAccSS4x8(A *byte, B *byte, C *int32) { vmadotAccSS4x8(A, B, C) }
