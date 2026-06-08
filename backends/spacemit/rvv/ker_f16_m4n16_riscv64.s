#include "textflag.h"

// Local RVV/Zvfh macros for the FP16 M4xN16 tile kernel. Go asm has no
// RVV/Zvfh mnemonics; encodings were generated with GNU as
// (-march=rv64gcv_zvfh) and objdump-verified.
#define VSETVLI_E32_M2_16        WORD $0x0d1ff057 // vsetvli zero, t6, e32, m2, ta, ma
#define VSETVLI_E16_M1_16        WORD $0x0c8ff057 // vsetvli zero, t6, e16, m1, ta, ma
#define VMV_V_I_V8_0             WORD $0x5e003457 // vmv.v.i v8,  0
#define VMV_V_I_V10_0            WORD $0x5e003557 // vmv.v.i v10, 0
#define VMV_V_I_V12_0            WORD $0x5e003657 // vmv.v.i v12, 0
#define VMV_V_I_V14_0            WORD $0x5e003757 // vmv.v.i v14, 0
#define VLE16_V0_A1              WORD $0x0205d007 // vle16.v  v0, (a1)
#define VLSE16_V2_A0_ZERO        WORD $0x0a055107 // vlse16.v v2,  (a0), zero
#define VLSE16_V4_T0_ZERO        WORD $0x0a02d207 // vlse16.v v4,  (t0), zero
#define VLSE16_V6_T1_ZERO        WORD $0x0a035307 // vlse16.v v6,  (t1), zero
#define VLSE16_V16_T2_ZERO       WORD $0x0a03d807 // vlse16.v v16, (t2), zero
#define VFWMACC_VV_V8_V2_V0      WORD $0xf2011457 // vfwmacc.vv v8,  v2,  v0
#define VFWMACC_VV_V10_V4_V0     WORD $0xf2021557 // vfwmacc.vv v10, v4,  v0
#define VFWMACC_VV_V12_V6_V0     WORD $0xf2031657 // vfwmacc.vv v12, v6,  v0
#define VFWMACC_VV_V14_V16_V0    WORD $0xf2081757 // vfwmacc.vv v14, v16, v0
#define VSE32_V8_A2              WORD $0x02066427 // vse32.v v8,  (a2)
#define VSE32_V10_A2             WORD $0x02066527 // vse32.v v10, (a2)
#define VSE32_V12_A2             WORD $0x02066627 // vse32.v v12, (a2)
#define VSE32_V14_A2             WORD $0x02066727 // vse32.v v14, (a2)

// func kernelF16M4N16(a, bp *uint16, c *float32, K, lda, ldc int64)
//
// 4(M) x 16(N) outer-product FP16 tile. Bp is packed as [K][16] fp16 words.
// Computes four rows of C with f32 accumulation/output, using Zvfh widening FMA:
//    C[4,16] += A[4,K] * Bp[K,16]
//
// Each A scalar is broadcast with an RVV strided load using zero stride, then
// accumulated with vfwmacc.vv. This avoids the scalar-FP vfwmacc.vf form, which
// traps on the K3 kernel path despite the advertised zfh/zvfh ISA string.
//
// Scalar regs used by encoded vector insns: a0=X10 a1=X11 a2=X12 a3=X13
// a4=X14 a5=X15 t0=X5 t1=X6 t2=X7 t6=X31.
TEXT ·kernelF16M4N16(SB), NOSPLIT, $0-48
	MOV	a+0(FP), X10
	MOV	bp+8(FP), X11
	MOV	c+16(FP), X12
	MOV	K+24(FP), X13
	MOV	lda+32(FP), X14
	MOV	ldc+40(FP), X15

	ADD	X14, X10, X5 // t0 = row1 pointer
	ADD	X14, X5, X6  // t1 = row2 pointer
	ADD	X14, X6, X7  // t2 = row3 pointer
	MOV	$16, X31      // t6 = vector length for N16 tile

	VSETVLI_E32_M2_16
	VMV_V_I_V8_0
	VMV_V_I_V10_0
	VMV_V_I_V12_0
	VMV_V_I_V14_0

loop:
	BEQZ	X13, store
	VSETVLI_E16_M1_16
	VLE16_V0_A1
	ADD	$32, X11, X11 // bp += 16 fp16 values

	VLSE16_V2_A0_ZERO
	ADD	$2, X10, X10
	VFWMACC_VV_V8_V2_V0

	VLSE16_V4_T0_ZERO
	ADD	$2, X5, X5
	VFWMACC_VV_V10_V4_V0

	VLSE16_V6_T1_ZERO
	ADD	$2, X6, X6
	VFWMACC_VV_V12_V6_V0

	VLSE16_V16_T2_ZERO
	ADD	$2, X7, X7
	VFWMACC_VV_V14_V16_V0

	ADD	$-1, X13, X13
	JMP	loop

store:
	VSETVLI_E32_M2_16
	VSE32_V8_A2
	ADD	X15, X12, X12
	VSE32_V10_A2
	ADD	X15, X12, X12
	VSE32_V12_A2
	ADD	X15, X12, X12
	VSE32_V14_A2
	RET
