// q8dot_amd64.s — AVX2 int8×float32 dot kernel

#include "textflag.h"

// func dotI8F32Asm(q []byte, x []float32) float32
// Requires len(q)==len(x) and len(q)%8==0; Go wrapper handles validation/fallback.
TEXT ·dotI8F32Asm(SB), NOSPLIT, $0-52
    MOVQ    q_base+0(FP), SI
    MOVQ    q_len+8(FP), CX
    MOVQ    x_base+24(FP), DI

    VXORPS  Y0, Y0, Y0

    TESTQ   CX, CX
    JZ      q8dot_reduce

q8dot_loop8:
    VPMOVSXBD (SI), Y1          // 8× int8 -> 8× int32
    VCVTDQ2PS Y1, Y1            // 8× int32 -> 8× float32
    VFMADD231PS (DI), Y1, Y0    // acc += q * x
    ADDQ    $8, SI
    ADDQ    $32, DI
    SUBQ    $8, CX
    JNZ     q8dot_loop8

q8dot_reduce:
    VEXTRACTF128 $1, Y0, X1
    VADDPS  X1, X0, X0
    VHADDPS X0, X0, X0
    VHADDPS X0, X0, X0
    VMOVSS  X0, ret+48(FP)
    VZEROUPPER
    RET
