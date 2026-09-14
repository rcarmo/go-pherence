// Four-output independent-column FMA. Each accumulator preserves ascending K.
#include "textflag.h"

TEXT ·fmaColumns4Asm(SB), NOSPLIT, $16-73
	STMXCSR 0(SP)
	MOVL 0(SP), AX
	TESTL $0xe040, AX
	JNZ bad
	MOVQ dst_base+0(FP), DI
	MOVQ dst_len+8(FP), AX
	SHRQ $2, AX                 // columns
	MOVQ x_base+24(FP), SI
	MOVQ weight_base+48(FP), DX
	MOVQ weight_len+56(FP), R8
	SHRQ $2, R8                 // reduction length
	MOVQ AX, R9
	SHLQ $2, R9                 // x row stride in bytes
	MOVQ R8, R10
	SHLQ $2, R10                // weight row stride in bytes
	MOVQ DX, BX
	ADDQ R10, BX                // weight row 1
	MOVQ BX, CX
	ADDQ R10, CX                // weight row 2
	MOVQ CX, R12
	ADDQ R10, R12               // weight row 3
	MOVQ R12, 8(SP)
	XORQ R11, R11               // output column
vector:
	MOVQ AX, R10
	SUBQ R11, R10
	CMPQ R10, $8
	JL tail
	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3
	LEAQ (SI)(R11*4), R12
	XORQ R13, R13
reduce8:
	CMPQ R13, R8
	JGE store8
	VMOVUPS (R12), Y4
	VBROADCASTSS (DX)(R13*4), Y5
	VFMADD231PS Y4, Y5, Y0
	VBROADCASTSS (BX)(R13*4), Y5
	VFMADD231PS Y4, Y5, Y1
	VBROADCASTSS (CX)(R13*4), Y5
	VFMADD231PS Y4, Y5, Y2
	MOVQ 8(SP), R10
	VBROADCASTSS (R10)(R13*4), Y5
	VFMADD231PS Y4, Y5, Y3
	ADDQ R9, R12
	INCQ R13
	JMP reduce8
store8:
	VMOVUPS Y0, (DI)(R11*4)
	LEAQ (DI)(AX*4), R10
	VMOVUPS Y1, (R10)(R11*4)
	LEAQ (R10)(AX*4), R10
	VMOVUPS Y2, (R10)(R11*4)
	LEAQ (R10)(AX*4), R10
	VMOVUPS Y3, (R10)(R11*4)
	ADDQ $8, R11
	JMP vector
tail:
	CMPQ R11, AX
	JGE done
	VXORPS X0, X0, X0
	VXORPS X1, X1, X1
	VXORPS X2, X2, X2
	VXORPS X3, X3, X3
	LEAQ (SI)(R11*4), R12
	XORQ R13, R13
reduce1:
	CMPQ R13, R8
	JGE store1
	VMOVSS (R12), X4
	VMOVSS (DX)(R13*4), X5
	VFMADD231SS X4, X5, X0
	VMOVSS (BX)(R13*4), X5
	VFMADD231SS X4, X5, X1
	VMOVSS (CX)(R13*4), X5
	VFMADD231SS X4, X5, X2
	MOVQ 8(SP), R10
	VMOVSS (R10)(R13*4), X5
	VFMADD231SS X4, X5, X3
	ADDQ R9, R12
	INCQ R13
	JMP reduce1
store1:
	VMOVSS X0, (DI)(R11*4)
	LEAQ (DI)(AX*4), R10
	VMOVSS X1, (R10)(R11*4)
	LEAQ (R10)(AX*4), R10
	VMOVSS X2, (R10)(R11*4)
	LEAQ (R10)(AX*4), R10
	VMOVSS X3, (R10)(R11*4)
	INCQ R11
	JMP tail
done:
	VZEROUPPER
	MOVB $1, ret+72(FP)
	RET
bad:
	MOVB $0, ret+72(FP)
	RET
