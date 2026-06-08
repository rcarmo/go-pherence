#include "textflag.h"

// Local RVV/Zvfh instruction macros. Go asm has no RVV/Zvfh mnemonics yet; keep
// the raw encodings named and colocated with the kernel that uses them. Encodings
// were generated with GNU as (-march=rv64gcv_zvfh) and objdump-verified.
#define VSETVLI_E32_M4_ZERO_TU_MU WORD $0x012072d7 // vsetvli t0, zero, e32, m4, tu, mu
#define VSETVLI_E16_M2_A2_TU_MU   WORD $0x009672d7 // vsetvli t0, a2,   e16, m2, tu, mu
#define VMV_V_I_V16_0             WORD $0x5e003857 // vmv.v.i  v16, 0
#define VMV_V_I_V8_0              WORD $0x5e003457 // vmv.v.i  v8,  0
#define VLE16_V0_A0               WORD $0x02055007 // vle16.v  v0, (a0)
#define VLE16_V8_A1               WORD $0x0205d407 // vle16.v  v8, (a1)
#define VFWMACC_VV_V16_V0_V8      WORD $0xf2801857 // vfwmacc.vv v16, v0, v8
#define VFREDUSUM_VS_V0_V16_V8    WORD $0x07041057 // vfredusum.vs v0, v16, v8
#define VFMV_F_S_FA0_V0           WORD $0x42001557 // vfmv.f.s fa0, v0

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
	VSETVLI_E32_M4_ZERO_TU_MU
	VMV_V_I_V16_0

loop:
	BEQZ	X12, done
	VSETVLI_E16_M2_A2_TU_MU
	VLE16_V0_A0
	VLE16_V8_A1
	VFWMACC_VV_V16_V0_V8 // f32 += f16*f16
	SLL	$1, X5, X6       // bytes = vl * sizeof(uint16)
	ADD	X6, X10, X10
	ADD	X6, X11, X11
	SUB	X5, X12, X12
	JMP	loop

done:
	// Reduce the full e32,m4 accumulator to fa0.
	VSETVLI_E32_M4_ZERO_TU_MU
	VMV_V_I_V8_0
	VFREDUSUM_VS_V0_V16_V8
	VFMV_F_S_FA0_V0
	MOVF	F10, ret+24(FP)
	RET
