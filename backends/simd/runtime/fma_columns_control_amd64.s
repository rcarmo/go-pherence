// Test shim, analogous to affineTestControl: restore controls before Go resumes.
#include "textflag.h"
TEXT ·fmaColumnsTestControl(SB), NOSPLIT, $88-81
	STMXCSR 80(SP)
	MOVL 80(SP), AX
	MOVL mask+72(FP), CX
	ORL CX, AX
	MOVL AX, 84(SP)
	LDMXCSR 84(SP)
	MOVQ dst_base+0(FP), AX
	MOVQ AX, 0(SP)
	MOVQ dst_len+8(FP), AX
	MOVQ AX, 8(SP)
	MOVQ dst_cap+16(FP), AX
	MOVQ AX, 16(SP)
	MOVQ x_base+24(FP), AX
	MOVQ AX, 24(SP)
	MOVQ x_len+32(FP), AX
	MOVQ AX, 32(SP)
	MOVQ x_cap+40(FP), AX
	MOVQ AX, 40(SP)
	MOVQ weight_base+48(FP), AX
	MOVQ AX, 48(SP)
	MOVQ weight_len+56(FP), AX
	MOVQ AX, 56(SP)
	MOVQ weight_cap+64(FP), AX
	MOVQ AX, 64(SP)
	CALL ·fmaColumnsAsm(SB)
	MOVBLZX 72(SP), AX
	LDMXCSR 80(SP)
	MOVB AX, ret+80(FP)
	RET
