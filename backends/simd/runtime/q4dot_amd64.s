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
