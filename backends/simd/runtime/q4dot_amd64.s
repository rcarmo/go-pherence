// q4dot_amd64.s — AVX2 uint4×float32 dot+sum kernels

#include "textflag.h"

// func dotU4F32LowAndSumAsm(q []byte, x []float32) (dot float32, sum float32)
// Requires len(q)==len(x) and len(q)%8==0; Go wrapper handles validation/fallback.
TEXT ·dotU4F32LowAndSumAsm(SB), NOSPLIT, $0-56
    MOVQ    q_base+0(FP), SI
    MOVQ    q_len+8(FP), CX
    MOVQ    x_base+24(FP), DI

    VXORPS  Y0, Y0, Y0          // dot accumulator
    VXORPS  Y2, Y2, Y2          // x sum accumulator
    VPCMPEQD Y15, Y15, Y15
    VPSRLD  $28, Y15, Y15       // dword mask 0x0000000f

    TESTQ   CX, CX
    JZ      q4low_reduce

q4low_loop8:
    VPMOVZXBD (SI), Y1          // 8× uint8 -> 8× uint32
    VPAND   Y15, Y1, Y1         // low nibble
    VCVTDQ2PS Y1, Y1            // 8× int32 -> 8× float32
    VMOVUPS (DI), Y3
    VFMADD231PS Y3, Y1, Y0      // dot += q * x
    VADDPS  Y3, Y2, Y2          // sum += x
    ADDQ    $8, SI
    ADDQ    $32, DI
    SUBQ    $8, CX
    JNZ     q4low_loop8

q4low_reduce:
    VEXTRACTF128 $1, Y0, X1
    VADDPS  X1, X0, X0
    VHADDPS X0, X0, X0
    VHADDPS X0, X0, X0

    VEXTRACTF128 $1, Y2, X3
    VADDPS  X3, X2, X2
    VHADDPS X2, X2, X2
    VHADDPS X2, X2, X2

    VMOVSS  X0, dot+48(FP)
    VMOVSS  X2, sum+52(FP)
    VZEROUPPER
    RET

// func dotU4F32HighAndSumAsm(q []byte, x []float32) (dot float32, sum float32)
// Requires len(q)==len(x) and len(q)%8==0; Go wrapper handles validation/fallback.
TEXT ·dotU4F32HighAndSumAsm(SB), NOSPLIT, $0-56
    MOVQ    q_base+0(FP), SI
    MOVQ    q_len+8(FP), CX
    MOVQ    x_base+24(FP), DI

    VXORPS  Y0, Y0, Y0          // dot accumulator
    VXORPS  Y2, Y2, Y2          // x sum accumulator
    VPCMPEQD Y15, Y15, Y15
    VPSRLD  $28, Y15, Y15       // dword mask 0x0000000f

    TESTQ   CX, CX
    JZ      q4high_reduce

q4high_loop8:
    VPMOVZXBD (SI), Y1          // 8× uint8 -> 8× uint32
    VPSRLD  $4, Y1, Y1
    VPAND   Y15, Y1, Y1         // high nibble
    VCVTDQ2PS Y1, Y1            // 8× int32 -> 8× float32
    VMOVUPS (DI), Y3
    VFMADD231PS Y3, Y1, Y0      // dot += q * x
    VADDPS  Y3, Y2, Y2          // sum += x
    ADDQ    $8, SI
    ADDQ    $32, DI
    SUBQ    $8, CX
    JNZ     q4high_loop8

q4high_reduce:
    VEXTRACTF128 $1, Y0, X1
    VADDPS  X1, X0, X0
    VHADDPS X0, X0, X0
    VHADDPS X0, X0, X0

    VEXTRACTF128 $1, Y2, X3
    VADDPS  X3, X2, X2
    VHADDPS X2, X2, X2
    VHADDPS X2, X2, X2

    VMOVSS  X0, dot+48(FP)
    VMOVSS  X2, sum+52(FP)
    VZEROUPPER
    RET

// func dotU4F32LowAndSumx4Asm(q []byte, x []float32, stride int) (dot0,sum0,dot1,sum1,dot2,sum2,dot3,sum3 float32)
// Requires len(q)==len(x group) and len(q)%8==0; Go wrapper validates shape.
TEXT ·dotU4F32LowAndSumx4Asm(SB), NOSPLIT, $0-88
    MOVQ    q_base+0(FP), SI
    MOVQ    q_len+8(FP), CX
    MOVQ    x_base+24(FP), DI
    MOVQ    stride+48(FP), R8
    SHLQ    $2, R8

    LEAQ    (DI)(R8*1), BX
    LEAQ    (BX)(R8*1), DX
    LEAQ    (DX)(R8*1), R9

    VXORPS  Y0, Y0, Y0      // dot0
    VXORPS  Y1, Y1, Y1      // sum0
    VXORPS  Y2, Y2, Y2      // dot1
    VXORPS  Y3, Y3, Y3      // sum1
    VXORPS  Y4, Y4, Y4      // dot2
    VXORPS  Y5, Y5, Y5      // sum2
    VXORPS  Y6, Y6, Y6      // dot3
    VXORPS  Y7, Y7, Y7      // sum3
    VPCMPEQD Y15, Y15, Y15
    VPSRLD  $28, Y15, Y15   // dword mask 0x0000000f

    TESTQ   CX, CX
    JZ      q4lowx4_reduce

q4lowx4_loop8:
    VPMOVZXBD (SI), Y8
    VPAND   Y15, Y8, Y8
    VCVTDQ2PS Y8, Y8

    VMOVUPS (DI), Y9
    VFMADD231PS Y9, Y8, Y0
    VADDPS  Y9, Y1, Y1

    VMOVUPS (BX), Y9
    VFMADD231PS Y9, Y8, Y2
    VADDPS  Y9, Y3, Y3

    VMOVUPS (DX), Y9
    VFMADD231PS Y9, Y8, Y4
    VADDPS  Y9, Y5, Y5

    VMOVUPS (R9), Y9
    VFMADD231PS Y9, Y8, Y6
    VADDPS  Y9, Y7, Y7

    ADDQ    $8, SI
    ADDQ    $32, DI
    ADDQ    $32, BX
    ADDQ    $32, DX
    ADDQ    $32, R9
    SUBQ    $8, CX
    JNZ     q4lowx4_loop8

q4lowx4_reduce:
    VEXTRACTF128 $1, Y0, X10
    VADDPS  X10, X0, X0
    VHADDPS X0, X0, X0
    VHADDPS X0, X0, X0

    VEXTRACTF128 $1, Y1, X10
    VADDPS  X10, X1, X1
    VHADDPS X1, X1, X1
    VHADDPS X1, X1, X1

    VEXTRACTF128 $1, Y2, X10
    VADDPS  X10, X2, X2
    VHADDPS X2, X2, X2
    VHADDPS X2, X2, X2

    VEXTRACTF128 $1, Y3, X10
    VADDPS  X10, X3, X3
    VHADDPS X3, X3, X3
    VHADDPS X3, X3, X3

    VEXTRACTF128 $1, Y4, X10
    VADDPS  X10, X4, X4
    VHADDPS X4, X4, X4
    VHADDPS X4, X4, X4

    VEXTRACTF128 $1, Y5, X10
    VADDPS  X10, X5, X5
    VHADDPS X5, X5, X5
    VHADDPS X5, X5, X5

    VEXTRACTF128 $1, Y6, X10
    VADDPS  X10, X6, X6
    VHADDPS X6, X6, X6
    VHADDPS X6, X6, X6

    VEXTRACTF128 $1, Y7, X10
    VADDPS  X10, X7, X7
    VHADDPS X7, X7, X7
    VHADDPS X7, X7, X7

    VMOVSS  X0, dot0+56(FP)
    VMOVSS  X1, sum0+60(FP)
    VMOVSS  X2, dot1+64(FP)
    VMOVSS  X3, sum1+68(FP)
    VMOVSS  X4, dot2+72(FP)
    VMOVSS  X5, sum2+76(FP)
    VMOVSS  X6, dot3+80(FP)
    VMOVSS  X7, sum3+84(FP)
    VZEROUPPER
    RET

// func dotU4F32HighAndSumx4Asm(q []byte, x []float32, stride int) (dot0,sum0,dot1,sum1,dot2,sum2,dot3,sum3 float32)
// Requires len(q)==len(x group) and len(q)%8==0; Go wrapper validates shape.
TEXT ·dotU4F32HighAndSumx4Asm(SB), NOSPLIT, $0-88
    MOVQ    q_base+0(FP), SI
    MOVQ    q_len+8(FP), CX
    MOVQ    x_base+24(FP), DI
    MOVQ    stride+48(FP), R8
    SHLQ    $2, R8

    LEAQ    (DI)(R8*1), BX
    LEAQ    (BX)(R8*1), DX
    LEAQ    (DX)(R8*1), R9

    VXORPS  Y0, Y0, Y0      // dot0
    VXORPS  Y1, Y1, Y1      // sum0
    VXORPS  Y2, Y2, Y2      // dot1
    VXORPS  Y3, Y3, Y3      // sum1
    VXORPS  Y4, Y4, Y4      // dot2
    VXORPS  Y5, Y5, Y5      // sum2
    VXORPS  Y6, Y6, Y6      // dot3
    VXORPS  Y7, Y7, Y7      // sum3
    VPCMPEQD Y15, Y15, Y15
    VPSRLD  $28, Y15, Y15   // dword mask 0x0000000f

    TESTQ   CX, CX
    JZ      q4highx4_reduce

q4highx4_loop8:
    VPMOVZXBD (SI), Y8
    VPSRLD  $4, Y8, Y8
    VPAND   Y15, Y8, Y8
    VCVTDQ2PS Y8, Y8

    VMOVUPS (DI), Y9
    VFMADD231PS Y9, Y8, Y0
    VADDPS  Y9, Y1, Y1

    VMOVUPS (BX), Y9
    VFMADD231PS Y9, Y8, Y2
    VADDPS  Y9, Y3, Y3

    VMOVUPS (DX), Y9
    VFMADD231PS Y9, Y8, Y4
    VADDPS  Y9, Y5, Y5

    VMOVUPS (R9), Y9
    VFMADD231PS Y9, Y8, Y6
    VADDPS  Y9, Y7, Y7

    ADDQ    $8, SI
    ADDQ    $32, DI
    ADDQ    $32, BX
    ADDQ    $32, DX
    ADDQ    $32, R9
    SUBQ    $8, CX
    JNZ     q4highx4_loop8

q4highx4_reduce:
    VEXTRACTF128 $1, Y0, X10
    VADDPS  X10, X0, X0
    VHADDPS X0, X0, X0
    VHADDPS X0, X0, X0

    VEXTRACTF128 $1, Y1, X10
    VADDPS  X10, X1, X1
    VHADDPS X1, X1, X1
    VHADDPS X1, X1, X1

    VEXTRACTF128 $1, Y2, X10
    VADDPS  X10, X2, X2
    VHADDPS X2, X2, X2
    VHADDPS X2, X2, X2

    VEXTRACTF128 $1, Y3, X10
    VADDPS  X10, X3, X3
    VHADDPS X3, X3, X3
    VHADDPS X3, X3, X3

    VEXTRACTF128 $1, Y4, X10
    VADDPS  X10, X4, X4
    VHADDPS X4, X4, X4
    VHADDPS X4, X4, X4

    VEXTRACTF128 $1, Y5, X10
    VADDPS  X10, X5, X5
    VHADDPS X5, X5, X5
    VHADDPS X5, X5, X5

    VEXTRACTF128 $1, Y6, X10
    VADDPS  X10, X6, X6
    VHADDPS X6, X6, X6
    VHADDPS X6, X6, X6

    VEXTRACTF128 $1, Y7, X10
    VADDPS  X10, X7, X7
    VHADDPS X7, X7, X7
    VHADDPS X7, X7, X7

    VMOVSS  X0, dot0+56(FP)
    VMOVSS  X1, sum0+60(FP)
    VMOVSS  X2, dot1+64(FP)
    VMOVSS  X3, sum1+68(FP)
    VMOVSS  X4, dot2+72(FP)
    VMOVSS  X5, sum2+76(FP)
    VMOVSS  X6, dot3+80(FP)
    VMOVSS  X7, sum3+84(FP)
    VZEROUPPER
    RET
