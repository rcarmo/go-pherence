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

// func q6KExpandCoeffAsm(block *[210]byte, coeff *[256]int16)
TEXT ·q6KExpandCoeffAsm(SB), NOSPLIT, $0-16
	MOVQ block+0(FP), SI
	LEAQ 128(SI), BX
	LEAQ 192(SI), DX
	MOVQ coeff+8(FP), DI
	MOVL $0x0f0f0f0f, AX
	VMOVD AX, X14
	VPBROADCASTD X14, Y14
	MOVL $0x03030303, AX
	VMOVD AX, X15
	VPBROADCASTD X15, Y15
	MOVL $0x20202020, AX
	VMOVD AX, X13
	VPBROADCASTD X13, Y13
	XORQ R9, R9
q6expand_half:
	XORQ R10, R10
q6expand_group:
	XORQ R11, R11
q6expand_chunk:
	MOVQ R10, R12
	ANDQ $1, R12
	SHLQ $5, R12
	MOVQ R11, AX
	SHLQ $4, AX
	ADDQ AX, R12
	VMOVDQU (SI)(R12*1), X0
	MOVQ R11, AX
	SHLQ $4, AX
	VMOVDQU (BX)(AX*1), X1
	CMPQ R10, $0
	JE q6expand_g0
	CMPQ R10, $1
	JE q6expand_g1
	CMPQ R10, $2
	JE q6expand_g2
	VPSRLW $4, X0, X0
	VPSRLW $6, X1, X1
	JMP q6expand_join
q6expand_g2:
	VPSRLW $4, X0, X0
	VPSRLW $4, X1, X1
	JMP q6expand_join
q6expand_g1:
	VPSRLW $2, X1, X1
	JMP q6expand_join
q6expand_g0:
q6expand_join:
	VPAND X14, X0, X0
	VPAND X15, X1, X1
	VPSLLW $4, X1, X1
	VPOR X1, X0, X0
	VPSUBB X13, X0, X0
	VPMOVSXBW X0, Y0
	MOVQ R10, R12
	SHLQ $1, R12
	ADDQ R11, R12
	MOVBQSX (DX)(R12*1), AX
	VMOVD AX, X2
	VPBROADCASTW X2, Y2
	VPMULLW Y2, Y0, Y0
	VMOVDQU Y0, (DI)
	ADDQ $32, DI
	INCQ R11
	CMPQ R11, $2
	JL q6expand_chunk
	INCQ R10
	CMPQ R10, $4
	JL q6expand_group
	ADDQ $64, SI
	ADDQ $32, BX
	ADDQ $8, DX
	INCQ R9
	CMPQ R9, $2
	JL q6expand_half
	VZEROUPPER
	RET

// func q6KBlockDotAsm(block *[210]byte, q8 *[256]int8) int32
// Expands Q6_K coefficients and consumes them immediately, preserving the
// eight-dword integer accumulation used by q6KCoeffDotAsm without a 512-byte
// coefficient temporary.
TEXT ·q6KBlockDotAsm(SB), NOSPLIT, $0-20
	MOVQ block+0(FP), SI
	LEAQ 128(SI), BX
	LEAQ 192(SI), DX
	MOVQ q8+8(FP), DI
	MOVL $0x0f0f0f0f, AX
	VMOVD AX, X14
	VPBROADCASTD X14, Y14
	MOVL $0x03030303, AX
	VMOVD AX, X15
	VPBROADCASTD X15, Y15
	MOVL $0x20202020, AX
	VMOVD AX, X13
	VPBROADCASTD X13, Y13
	VPXOR Y3, Y3, Y3
	XORQ R9, R9
q6blockdot_half:
	XORQ R10, R10
q6blockdot_group:
	XORQ R11, R11
q6blockdot_chunk:
	MOVQ R10, R12
	ANDQ $1, R12
	SHLQ $5, R12
	MOVQ R11, AX
	SHLQ $4, AX
	ADDQ AX, R12
	VMOVDQU (SI)(R12*1), X0
	MOVQ R11, AX
	SHLQ $4, AX
	VMOVDQU (BX)(AX*1), X1
	CMPQ R10, $0
	JE q6blockdot_g0
	CMPQ R10, $1
	JE q6blockdot_g1
	CMPQ R10, $2
	JE q6blockdot_g2
	VPSRLW $4, X0, X0
	VPSRLW $6, X1, X1
	JMP q6blockdot_join
q6blockdot_g2:
	VPSRLW $4, X0, X0
	VPSRLW $4, X1, X1
	JMP q6blockdot_join
q6blockdot_g1:
	VPSRLW $2, X1, X1
	JMP q6blockdot_join
q6blockdot_g0:
q6blockdot_join:
	VPAND X14, X0, X0
	VPAND X15, X1, X1
	VPSLLW $4, X1, X1
	VPOR X1, X0, X0
	VPSUBB X13, X0, X0
	VPMOVSXBW X0, Y0
	MOVQ R10, R12
	SHLQ $1, R12
	ADDQ R11, R12
	MOVBQSX (DX)(R12*1), AX
	VMOVD AX, X2
	VPBROADCASTW X2, Y2
	VPMULLW Y2, Y0, Y0
	VPMOVSXBW (DI), Y1
	VPMADDWD Y0, Y1, Y1
	VPADDD Y1, Y3, Y3
	ADDQ $16, DI
	INCQ R11
	CMPQ R11, $2
	JL q6blockdot_chunk
	INCQ R10
	CMPQ R10, $4
	JL q6blockdot_group
	ADDQ $64, SI
	ADDQ $32, BX
	ADDQ $8, DX
	INCQ R9
	CMPQ R9, $2
	JL q6blockdot_half
	VEXTRACTI128 $1, Y3, X1
	VPADDD X1, X3, X3
	VPSHUFD $0x4e, X3, X1
	VPADDD X1, X3, X3
	VPSHUFD $0xb1, X3, X1
	VPADDD X1, X3, X3
	VMOVD X3, ret+16(FP)
	VZEROUPPER
	RET
