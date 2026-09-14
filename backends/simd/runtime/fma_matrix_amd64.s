#include "textflag.h"

// Check control state, overwrite C with +0, and invoke existing serial-K SGEMM
// in one assembly call. All shape/alias/finiteness/ISA checks happen in Go.
TEXT ·fmaMatrixAsm(SB), NOSPLIT, $88-97
	STMXCSR 80(SP)
	MOVL 80(SP), AX
	TESTL $0xe040, AX
	JNZ bad
	MOVQ dst_base+0(FP), DI
	MOVQ dst_len+8(FP), CX
	XORL AX, AX
zero:
	MOVL AX, (DI)
	ADDQ $4, DI
	DECQ CX
	JNZ zero
	MOVQ m+72(FP), AX
	MOVQ AX, 0(SP)
	MOVQ n+80(FP), AX
	MOVQ AX, 8(SP)
	MOVQ AX, 64(SP)
	MOVQ AX, 72(SP)
	MOVQ k+88(FP), AX
	MOVQ AX, 16(SP)
	MOVQ AX, 56(SP)
	MOVL $0x3f800000, 24(SP)
	MOVQ a_base+24(FP), AX
	MOVQ AX, 32(SP)
	MOVQ b_base+48(FP), AX
	MOVQ AX, 40(SP)
	MOVQ dst_base+0(FP), AX
	MOVQ AX, 48(SP)
	CALL ·SgemmNN(SB)
	MOVB $1, ret+96(FP)
	RET
bad:
	MOVB $0, ret+96(FP)
	RET
