#include "textflag.h"
// Test shim changes MXCSR only across the guarded assembly call, never Go.
TEXT ·fmaMatrixTestControl(SB), NOSPLIT, $112-105
	STMXCSR 104(SP)
	MOVL 104(SP), AX
	MOVL mask+96(FP), CX
	ORL CX, AX
	MOVL AX, 108(SP)
	LDMXCSR 108(SP)
	MOVQ dst_base+0(FP), AX
	MOVQ AX, 0(SP)
	MOVQ dst_len+8(FP), AX
	MOVQ AX, 8(SP)
	MOVQ dst_cap+16(FP), AX
	MOVQ AX, 16(SP)
	MOVQ a_base+24(FP), AX
	MOVQ AX, 24(SP)
	MOVQ a_len+32(FP), AX
	MOVQ AX, 32(SP)
	MOVQ a_cap+40(FP), AX
	MOVQ AX, 40(SP)
	MOVQ b_base+48(FP), AX
	MOVQ AX, 48(SP)
	MOVQ b_len+56(FP), AX
	MOVQ AX, 56(SP)
	MOVQ b_cap+64(FP), AX
	MOVQ AX, 64(SP)
	MOVQ m+72(FP), AX
	MOVQ AX, 72(SP)
	MOVQ n+80(FP), AX
	MOVQ AX, 80(SP)
	MOVQ k+88(FP), AX
	MOVQ AX, 88(SP)
	CALL ·fmaMatrixAsm(SB)
	MOVBLZX 96(SP), AX
	LDMXCSR 104(SP)
	MOVB AX, ret+104(FP)
	RET
