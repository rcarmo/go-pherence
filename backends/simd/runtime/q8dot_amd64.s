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

// func dotI8F32x4Asm(q []byte, x []float32, stride int) (float32, float32, float32, float32)
// Requires len(q)==len(x row) and len(q)%8==0; Go wrapper validates shape.
TEXT ·dotI8F32x4Asm(SB), NOSPLIT, $0-72
    MOVQ    q_base+0(FP), SI
    MOVQ    q_len+8(FP), CX
    MOVQ    x_base+24(FP), DI
    MOVQ    stride+48(FP), R8
    SHLQ    $2, R8              // stride in bytes

    LEAQ    (DI)(R8*1), BX      // x row 1
    LEAQ    (BX)(R8*1), DX      // x row 2
    LEAQ    (DX)(R8*1), R9      // x row 3

    VXORPS  Y0, Y0, Y0          // sum row 0
    VXORPS  Y1, Y1, Y1          // sum row 1
    VXORPS  Y2, Y2, Y2          // sum row 2
    VXORPS  Y3, Y3, Y3          // sum row 3

    TESTQ   CX, CX
    JZ      q8dotx4_reduce

q8dotx4_loop8:
    VPMOVSXBD (SI), Y4          // 8× int8 -> 8× int32
    VCVTDQ2PS Y4, Y4            // 8× int32 -> 8× float32
    VFMADD231PS (DI), Y4, Y0
    VFMADD231PS (BX), Y4, Y1
    VFMADD231PS (DX), Y4, Y2
    VFMADD231PS (R9), Y4, Y3
    ADDQ    $8, SI
    ADDQ    $32, DI
    ADDQ    $32, BX
    ADDQ    $32, DX
    ADDQ    $32, R9
    SUBQ    $8, CX
    JNZ     q8dotx4_loop8

q8dotx4_reduce:
    VEXTRACTF128 $1, Y0, X4
    VADDPS  X4, X0, X0
    VHADDPS X0, X0, X0
    VHADDPS X0, X0, X0

    VEXTRACTF128 $1, Y1, X4
    VADDPS  X4, X1, X1
    VHADDPS X1, X1, X1
    VHADDPS X1, X1, X1

    VEXTRACTF128 $1, Y2, X4
    VADDPS  X4, X2, X2
    VHADDPS X2, X2, X2
    VHADDPS X2, X2, X2

    VEXTRACTF128 $1, Y3, X4
    VADDPS  X4, X3, X3
    VHADDPS X3, X3, X3
    VHADDPS X3, X3, X3

    VMOVSS  X0, ret+56(FP)
    VMOVSS  X1, ret+60(FP)
    VMOVSS  X2, ret+64(FP)
    VMOVSS  X3, ret+68(FP)
    VZEROUPPER
    RET
