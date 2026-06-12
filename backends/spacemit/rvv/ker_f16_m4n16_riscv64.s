#include "textflag.h"
#include "../ime2/ime2_isa.h"

// func kernelF16M4N16(a, bp *uint16, c *float32, K, lda, ldc int64)
//
// 4(M) x 16(N) outer-product FP16 tile. Bp is packed as [K][16] fp16 words.
// Computes four rows of C with f32 accumulation/output, using Zvfh widening FMA:
//    C[4,16] += A[4,K] * Bp[K,16]
//
// Each A scalar is loaded as a halfword into an integer register and broadcast
// with vmv.v.x, then accumulated with vfwmacc.vv. This avoids vfwmacc.vf, which
// traps on the K3 kernel path, while also avoiding the slower zero-stride vlse16
// broadcast used by the first correctness kernel.
//
// Scalar regs used by encoded vector insns: a0=X10 a1=X11 a2=X12 a3=X13
// a4=X14 a5=X15 t0=X5 t1=X6 t2=X7 t3=X28 t4=X29 t5=X30 t6=X31.
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

	K3_VSETVLI_E32_M2_16
	K3_VMV_V_I_V8_0
	K3_VMV_V_I_V10_0
	K3_VMV_V_I_V12_0
	K3_VMV_V_I_V14_0

loop:
	BEQZ	X13, store
	K3_VSETVLI_E16_M1_16
	K3_VLE16_V0_A1
	ADD	$32, X11, X11 // bp += 16 fp16 values

	K3_LHU_T3_A0
	ADD	$2, X10, X10
	K3_VMV_V_X_V4_T3
	K3_VFWMACC_VV_V8_V4_V0

	K3_LHU_T4_T0
	ADD	$2, X5, X5
	K3_VMV_V_X_V4_T4
	K3_VFWMACC_VV_V10_V4_V0

	K3_LHU_T5_T1
	ADD	$2, X6, X6
	K3_VMV_V_X_V4_T5
	K3_VFWMACC_VV_V12_V4_V0

	K3_LHU_T3_T2
	ADD	$2, X7, X7
	K3_VMV_V_X_V4_T3
	K3_VFWMACC_VV_V14_V4_V0

	ADD	$-1, X13, X13
	JMP	loop

store:
	K3_VSETVLI_E32_M2_16
	K3_VSE32_V8_A2
	ADD	X15, X12, X12
	K3_VSE32_V10_A2
	ADD	X15, X12, X12
	K3_VSE32_V12_A2
	ADD	X15, X12, X12
	K3_VSE32_V14_A2
	RET
