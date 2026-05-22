//go:build riscv64

// vec_riscv64.s — basic RISC-V RVV elementwise kernels.

#include "textflag.h"

// Shared RVV encodings from GNU as/objdump:
//   vsetvli a3,a2,e32,m1,ta,ma  0x0d0676d7
//   vle32.v v0,(a0)             0x02056007
//   vle32.v v1,(a1)             0x0205e087
//   vfadd.vv v2,v0,v1           0x02009157
//   vfmul.vv v2,v0,v1           0x92009157
//   vfmul.vf v2,v0,fa0          0x92055157
//   vfmacc.vf v0,fa0,v1         0xb2155057
//   vse32.v v2,(a4)             0x02076127
//   vse32.v v0,(a4)             0x02076027

// func vecAddAsm(dst, a, b []float32)
TEXT ·vecAddAsm(SB), NOSPLIT, $0-72
	MOV	dst_base+0(FP), X14
	MOV	a_base+24(FP), X10
	MOV	a_len+32(FP), X12
	MOV	b_base+48(FP), X11
	BEQZ	X12, vec_add_done
vec_add_loop:
	WORD	$0x0d0676d7
	WORD	$0x02056007
	WORD	$0x0205e087
	WORD	$0x02009157
	WORD	$0x02076127
	SLLI	$2, X13, X15
	ADD	X15, X10, X10
	ADD	X15, X11, X11
	ADD	X15, X14, X14
	SUB	X13, X12, X12
	BNEZ	X12, vec_add_loop
vec_add_done:
	RET

// func vecMulAsm(dst, a, b []float32)
TEXT ·vecMulAsm(SB), NOSPLIT, $0-72
	MOV	dst_base+0(FP), X14
	MOV	a_base+24(FP), X10
	MOV	a_len+32(FP), X12
	MOV	b_base+48(FP), X11
	BEQZ	X12, vec_mul_done
vec_mul_loop:
	WORD	$0x0d0676d7
	WORD	$0x02056007
	WORD	$0x0205e087
	WORD	$0x92009157
	WORD	$0x02076127
	SLLI	$2, X13, X15
	ADD	X15, X10, X10
	ADD	X15, X11, X11
	ADD	X15, X14, X14
	SUB	X13, X12, X12
	BNEZ	X12, vec_mul_loop
vec_mul_done:
	RET

// func vecScaleAsm(dst, a []float32, scale float32)
TEXT ·vecScaleAsm(SB), NOSPLIT, $0-52
	MOV	dst_base+0(FP), X14
	MOV	a_base+24(FP), X10
	MOV	a_len+32(FP), X12
	MOVF	scale+48(FP), F10
	BEQZ	X12, vec_scale_done
vec_scale_loop:
	WORD	$0x0d0676d7
	WORD	$0x02056007
	WORD	$0x92055157
	WORD	$0x02076127
	SLLI	$2, X13, X15
	ADD	X15, X10, X10
	ADD	X15, X14, X14
	SUB	X13, X12, X12
	BNEZ	X12, vec_scale_loop
vec_scale_done:
	RET

// func vecScaleAddAsm(dst, a, b []float32, scale float32)
TEXT ·vecScaleAddAsm(SB), NOSPLIT, $0-76
	MOV	dst_base+0(FP), X14
	MOV	a_base+24(FP), X10
	MOV	a_len+32(FP), X12
	MOV	b_base+48(FP), X11
	MOVF	scale+72(FP), F10
	BEQZ	X12, vec_scale_add_done
vec_scale_add_loop:
	WORD	$0x0d0676d7
	WORD	$0x02056007
	WORD	$0x0205e087
	WORD	$0xb2155057
	WORD	$0x02076027
	SLLI	$2, X13, X15
	ADD	X15, X10, X10
	ADD	X15, X11, X11
	ADD	X15, X14, X14
	SUB	X13, X12, X12
	BNEZ	X12, vec_scale_add_loop
vec_scale_add_done:
	RET

// RVV encodings for rmsNormScaleAsm:
//   vfmul.vf v2,v2,fa0          0x92255157
//   vse32.v v2,(a0)             0x02056127

// func rmsNormScaleAsm(x, w []float32, scale float32)
TEXT ·rmsNormScaleAsm(SB), NOSPLIT, $0-52
	MOV	x_base+0(FP), X10
	MOV	x_len+8(FP), X12
	MOV	w_base+24(FP), X11
	MOVF	scale+48(FP), F10
	BEQZ	X12, rms_norm_scale_done
rms_norm_scale_loop:
	WORD	$0x0d0676d7           // vsetvli a3,a2,e32,m1,ta,ma
	WORD	$0x02056007           // vle32.v v0,(a0)
	WORD	$0x0205e087           // vle32.v v1,(a1)
	WORD	$0x92009157           // vfmul.vv v2,v0,v1
	WORD	$0x92255157           // vfmul.vf v2,v2,fa0
	WORD	$0x02056127           // vse32.v v2,(a0)
	SLLI	$2, X13, X14
	ADD	X14, X10, X10
	ADD	X14, X11, X11
	SUB	X13, X12, X12
	BNEZ	X12, rms_norm_scale_loop
rms_norm_scale_done:
	RET
