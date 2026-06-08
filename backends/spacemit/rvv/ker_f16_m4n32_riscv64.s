#include "textflag.h"
#include "../ime2/k3_isa.h"

// func kernelF16M4N32(a, bp *uint16, c *float32, K, lda, ldc int64)
//
// 4(M) x 32(N) outer-product FP16 tile. Bp is packed as [K][32] fp16 words.
// This is the X100/VLEN=256-friendly tile shape: e16,m2 loads 32 fp16 B values
// and e32,m4 stores 32 f32 accumulators per output row.
TEXT ·kernelF16M4N32(SB), NOSPLIT, $0-48
	MOV	a+0(FP), X10
	MOV	bp+8(FP), X11
	MOV	c+16(FP), X12
	MOV	K+24(FP), X13
	MOV	lda+32(FP), X14
	MOV	ldc+40(FP), X15

	ADD	X14, X10, X5 // t0 = row1 pointer
	ADD	X14, X5, X6  // t1 = row2 pointer
	ADD	X14, X6, X7  // t2 = row3 pointer
	MOV	$32, X31      // t6 = vector length for N32 tile

	K3_VSETVLI_E32_M4_32
	K3_VMV_V_I_V8_0
	K3_VMV_V_I_V12_0
	K3_VMV_V_I_V16_0
	K3_VMV_V_I_V20_0

loop:
	BEQZ	X13, store
	K3_VSETVLI_E16_M2_32
	K3_VLE16_V0_A1
	ADD	$64, X11, X11 // bp += 32 fp16 values

	K3_LHU_T3_A0
	ADD	$2, X10, X10
	K3_VMV_V_X_V4_T3
	K3_VFWMACC_VV_V8_V4_V0

	K3_LHU_T4_T0
	ADD	$2, X5, X5
	K3_VMV_V_X_V4_T4
	K3_VFWMACC_VV_V12_V4_V0

	K3_LHU_T5_T1
	ADD	$2, X6, X6
	K3_VMV_V_X_V4_T5
	K3_VFWMACC_VV_V16_V4_V0

	K3_LHU_T3_T2
	ADD	$2, X7, X7
	K3_VMV_V_X_V4_T3
	K3_VFWMACC_VV_V20_V4_V0

	ADD	$-1, X13, X13
	JMP	loop

store:
	K3_VSETVLI_E32_M4_32
	K3_VSE32_V8_A2
	ADD	X15, X12, X12
	K3_VSE32_V12_A2
	ADD	X15, X12, X12
	K3_VSE32_V16_A2
	ADD	X15, X12, X12
	K3_VSE32_V20_A2
	RET
