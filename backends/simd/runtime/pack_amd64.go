package simd

import "golang.org/x/sys/cpu"

//go:noescape
func packBNTAsm(
	b0, b1, b2, b3, b4, b5, b6, b7,
	b8, b9, b10, b11, b12, b13, b14, b15 uintptr,
	k int, bp uintptr)

// Match the runtime GEMM feature gate, including for direct packing calls.
var hasAvxPack = cpu.X86.HasAVX2 && cpu.X86.HasFMA

const hasNeonPack = false
