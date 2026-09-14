//go:build amd64

#include "textflag.h"

// func affineF32Asm(x []float32, scale, shift float32) bool
// Input validation/ISA admission in Go. Exact in-place 8-wide FMA + scalar tail.
// Leave MXCSR controls untouched. DAZ/FTZ/rounding affect tiny/midpoint values.
TEXT ·affineF32Asm(SB), NOSPLIT, $8-33
    STMXCSR 0(SP)
    MOVL 0(SP), AX
    TESTL $0xe040, AX             // rounding mode, FTZ, DAZ
    JNZ unsupported
    MOVQ x_base+0(FP), SI
    MOVQ x_len+8(FP), CX
    VBROADCASTSS scale+24(FP), Y1
    VBROADCASTSS shift+28(FP), Y2
loop:
    CMPQ CX, $8
    JL tail
    VMOVUPS (SI), Y0
    VFMADD213PS Y2, Y1, Y0        // Y0 = (Y1 * Y0) + Y2
    VMOVUPS Y0, (SI)
    ADDQ $32, SI
    SUBQ $8, CX
    JMP loop
tail:
    TESTQ CX, CX
    JZ done
scalar:
    VMOVSS (SI), X0
    VFMADD213SS X2, X1, X0
    VMOVSS X0, (SI)
    ADDQ $4, SI
    DECQ CX
    JNZ scalar
done:
    VZEROUPPER
    MOVB $1, ret+32(FP)
    RET
unsupported:
    MOVB $0, ret+32(FP)
    RET
