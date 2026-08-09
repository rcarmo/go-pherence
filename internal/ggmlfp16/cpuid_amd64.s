//go:build amd64

#include "textflag.h"

TEXT ·cpuHasF16C(SB), NOSPLIT, $0-1
	MOVQ BX, R8
	MOVL $1, AX
	CPUID
	SHRL $29, CX
	ANDL $1, CX
	MOVB CL, ret+0(FP)
	MOVQ R8, BX
	RET
