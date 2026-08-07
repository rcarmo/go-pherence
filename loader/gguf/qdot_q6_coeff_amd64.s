//go:build amd64

#include "textflag.h"

// func q6KCoeffDotAsm(q8 *[256]int8, coeff *[256]int16) int32
TEXT ·q6KCoeffDotAsm(SB), NOSPLIT, $0-20
    MOVQ q8+0(FP), SI
    MOVQ coeff+8(FP), DI
    MOVQ $16, CX
    VPXOR Y0, Y0, Y0
q6coeff_loop:
    VPMOVSXBW (SI), Y1
    VMOVDQU (DI), Y2
    VPMADDWD Y2, Y1, Y1
    VPADDD Y1, Y0, Y0
    ADDQ $16, SI
    ADDQ $32, DI
    DECQ CX
    JNZ q6coeff_loop
    VEXTRACTI128 $1, Y0, X1
    VPADDD X1, X0, X0
    VPSHUFD $0x4e, X0, X1
    VPADDD X1, X0, X0
    VPSHUFD $0xb1, X0, X1
    VPADDD X1, X0, X0
    VMOVD X0, ret+16(FP)
    VZEROUPPER
    RET

// func q6KCoeffDot8Asm(q8 *[256]int8, coeff *[256]int16, out *[8]int32)
TEXT ·q6KCoeffDot8Asm(SB), NOSPLIT, $0-24
    MOVQ q8+0(FP), SI
    MOVQ coeff+8(FP), DI
    MOVQ out+16(FP), DX
    MOVQ $16, CX
    VPXOR Y0, Y0, Y0
q6coeff8_loop:
    VPMOVSXBW (SI), Y1
    VMOVDQU (DI), Y2
    VPMADDWD Y2, Y1, Y1
    VPADDD Y1, Y0, Y0
    ADDQ $16, SI
    ADDQ $32, DI
    DECQ CX
    JNZ q6coeff8_loop
    VMOVDQU Y0, (DX)
    VZEROUPPER
    RET
