// q6_i8dot_amd64.s — AVX2 int8×int16 dot for DiffusionGemma Q6_K×Q8_K LM head

#include "textflag.h"

// func q6KBlockCoeffISumAsm(q8 []int8, coeff *[256]int16) int32
TEXT ·q6KBlockCoeffISumAsm(SB), NOSPLIT, $0-36
    MOVQ q8_base+0(FP), SI
    MOVQ coeff+24(FP), DI
    MOVQ $16, CX

    VPXOR Y0, Y0, Y0          // 8 x int32 accumulator

q6blk_loop:
    VPMOVSXBW (SI), Y1        // 16 x int8 -> 16 x int16
    VMOVDQU (DI), Y2          // 16 x int16 coeffs
    VPMADDWD Y2, Y1, Y1       // adjacent int16 products -> 8 x int32
    VPADDD Y1, Y0, Y0
    ADDQ $16, SI
    ADDQ $32, DI
    DECQ CX
    JNZ q6blk_loop

    VEXTRACTI128 $1, Y0, X1
    VPADDD X1, X0, X0         // 4 x int32
    VPSHUFD $0x4e, X0, X1
    VPADDD X1, X0, X0         // pair sums
    VPSHUFD $0xb1, X0, X1
    VPADDD X1, X0, X0         // final in lane 0
    VMOVD X0, ret+32(FP)
    VZEROUPPER
    RET
