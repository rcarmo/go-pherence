#include "textflag.h"
#include "../ime2/k3_isa.h"

// func dotF16(a, b *uint16, n int64) float32
// RVV/Zvfh fp16 dot product with fp32 accumulation.
//
// Scalar regs used by encoded vector insns: a0=X10 a1=X11 a2=X12 t0=X5.
// FP layout: a+0(FP)[8] b+8(FP)[8] n+16(FP)[8] ret+24(FP)[4]
TEXT ·dotF16(SB), NOSPLIT, $0-28
	MOV	a+0(FP),  X10
	MOV	b+8(FP),  X11
	MOV	n+16(FP), X12

	// Accumulator v16 (e32,m4 group v16..v19) = 0.0f.
	K3_VSETVLI_E32_M4_ZERO_TU_MU
	K3_VMV_V_I_V16_0

loop:
	BEQZ	X12, done
	K3_VSETVLI_E16_M2_A2_TU_MU
	K3_VLE16_V0_A0
	K3_VLE16_V8_A1
	K3_VFWMACC_VV_V16_V0_V8 // f32 += f16*f16
	SLL	$1, X5, X6          // bytes = vl * sizeof(uint16)
	ADD	X6, X10, X10
	ADD	X6, X11, X11
	SUB	X5, X12, X12
	JMP	loop

done:
	// Reduce the full e32,m4 accumulator to fa0.
	K3_VSETVLI_E32_M4_ZERO_TU_MU
	K3_VMV_V_I_V8_0
	K3_VFREDUSUM_VS_V0_V16_V8
	K3_VFMV_F_S_FA0_V0
	MOVF	F10, ret+24(FP)
	RET
