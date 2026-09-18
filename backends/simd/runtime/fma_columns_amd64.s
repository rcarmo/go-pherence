// Independent-column FMA: no horizontal reduction or reassociation.
#include "textflag.h"

TEXT ·fmaColumnsAsm(SB), NOSPLIT, $8-73
	STMXCSR 0(SP)
	MOVL 0(SP), AX
	TESTL $0xe040, AX
	JNZ bad
	MOVQ dst_base+0(FP), DI
	MOVQ dst_len+8(FP), CX
	MOVQ x_base+24(FP), SI
	MOVQ weight_base+48(FP), DX
	MOVQ weight_len+56(FP), R8
	MOVQ CX, R9
	SHLQ $2, R9
	XORQ R10, R10
vector:
	CMPQ CX, $8
	JL tail
	VXORPS Y0, Y0, Y0
	LEAQ (SI)(R10*4), R11
	XORQ AX, AX
reduce8:
	CMPQ AX, R8
	JGE store8
	VMOVUPS (R11), Y1
	VBROADCASTSS (DX)(AX*4), Y2
	VFMADD231PS Y1, Y2, Y0
	ADDQ R9, R11
	INCQ AX
	JMP reduce8
store8:
	VMOVUPS Y0, (DI)(R10*4)
	ADDQ $8, R10
	SUBQ $8, CX
	JMP vector
tail:
	TESTQ CX, CX
	JE done
	VXORPS X0, X0, X0
	LEAQ (SI)(R10*4), R11
	XORQ AX, AX
reduce1:
	CMPQ AX, R8
	JGE store1
	VMOVSS (R11), X1
	VMOVSS (DX)(AX*4), X2
	VFMADD231SS X1, X2, X0
	ADDQ R9, R11
	INCQ AX
	JMP reduce1
store1:
	VMOVSS X0, (DI)(R10*4)
	INCQ R10
	DECQ CX
	JMP tail
done:
	VZEROUPPER
	MOVB $1, ret+72(FP)
	RET
bad:
	MOVB $0, ret+72(FP)
	RET
