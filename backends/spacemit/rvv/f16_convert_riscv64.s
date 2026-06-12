#include "textflag.h"
#include "../ime2/ime2_isa.h"

// func f32ToF16RVV(src *float32, dst *uint16, n int)
// Vector narrowing conversion: f32 -> IEEE-754 fp16 bits.
// Scalar regs: a0=X10 src, a1=X11 dst, a2=X12 n, t0=X5 vl, t1=X6 bytes.
TEXT ·f32ToF16RVV(SB), NOSPLIT, $0-24
	MOV	src+0(FP), X10
	MOV	dst+8(FP), X11
	MOV	n+16(FP), X12
loop:
	BEQZ	X12, done
	K3_VSETVLI_E32_M4_A2_TA_MA
	K3_VLE32_V8_A0
	SLL	$2, X5, X6
	ADD	X6, X10, X10
	K3_VSETVLI_E16_M2_A2_TA_MA
	K3_VFNCVT_F_F_W_V0_V8
	K3_VSE16_V0_A1
	SLL	$1, X5, X6
	ADD	X6, X11, X11
	SUB	X5, X12, X12
	JMP	loop
done:
	RET
