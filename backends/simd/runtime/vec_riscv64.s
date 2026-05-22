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

// RVV encodings for toBF16Asm:
//   vsrl.vi v0,v0,16             0xa2083057
//   vsll.vi v0,v0,16             0x96083057
//   vse32.v v0,(a0)              0x02056027

// func toBF16Asm(x []float32)
TEXT ·toBF16Asm(SB), NOSPLIT, $0-24
	MOV	x_base+0(FP), X10
	MOV	x_len+8(FP), X12
	BEQZ	X12, to_bf16_done
to_bf16_loop:
	WORD	$0x0d0676d7           // vsetvli a3,a2,e32,m1,ta,ma
	WORD	$0x02056007           // vle32.v v0,(a0)
	WORD	$0xa2083057           // vsrl.vi v0,v0,16
	WORD	$0x96083057           // vsll.vi v0,v0,16
	WORD	$0x02056027           // vse32.v v0,(a0)
	SLLI	$2, X13, X14
	ADD	X14, X10, X10
	SUB	X13, X12, X12
	BNEZ	X12, to_bf16_loop
to_bf16_done:
	RET

// RVV encodings for BF16 widen/narrow:
//   vle16.v v0,(a1)              0x0205d007
//   vzext.vf2 v2,v0              0x4a032157
//   vsll.vi v2,v2,16             0x96283157
//   vnsrl.wi v2,v0,16            0xb2083157
//   vse16.v v2,(a0)              0x02055127

// func bf16WidenToF32Asm(dst []float32, src []uint16)
TEXT ·bf16WidenToF32Asm(SB), NOSPLIT, $0-48
	MOV	dst_base+0(FP), X10
	MOV	src_base+24(FP), X11
	MOV	src_len+32(FP), X12
	BEQZ	X12, bf16_widen_done
bf16_widen_loop:
	WORD	$0x0d0676d7           // vsetvli a3,a2,e32,m1,ta,ma
	WORD	$0x0205d007           // vle16.v v0,(a1)
	WORD	$0x4a032157           // vzext.vf2 v2,v0
	WORD	$0x96283157           // vsll.vi v2,v2,16
	WORD	$0x02056127           // vse32.v v2,(a0)
	SLLI	$2, X13, X14
	ADD	X14, X10, X10
	SLLI	$1, X13, X14
	ADD	X14, X11, X11
	SUB	X13, X12, X12
	BNEZ	X12, bf16_widen_loop
bf16_widen_done:
	RET

// func bf16NarrowFromF32Asm(dst []uint16, src []float32)
TEXT ·bf16NarrowFromF32Asm(SB), NOSPLIT, $0-48
	MOV	dst_base+0(FP), X10
	MOV	src_base+24(FP), X11
	MOV	src_len+32(FP), X12
	BEQZ	X12, bf16_narrow_done
bf16_narrow_loop:
	WORD	$0x0d0676d7           // vsetvli a3,a2,e32,m1,ta,ma
	WORD	$0x0205e007           // vle32.v v0,(a1)
	WORD	$0xb2083157           // vnsrl.wi v2,v0,16
	WORD	$0x02055127           // vse16.v v2,(a0)
	SLLI	$1, X13, X14
	ADD	X14, X10, X10
	SLLI	$2, X13, X14
	ADD	X14, X11, X11
	SUB	X13, X12, X12
	BNEZ	X12, bf16_narrow_loop
bf16_narrow_done:
	RET

// RVV encodings for BF16 add:
//   vle16.v v1,(a2)              0x02065087
//   vzext.vf2 v4,v1              0x4a132257
//   vsll.vi v4,v4,16             0x96483257
//   vfadd.vv v6,v2,v4            0x02221357
//   vnsrl.wi v8,v6,16            0xb2683457
//   vse16.v v8,(a0)              0x02055427

// func bf16VecAddAsm(dst, a, b []uint16)
TEXT ·bf16VecAddAsm(SB), NOSPLIT, $0-72
	MOV	dst_base+0(FP), X10
	MOV	a_base+24(FP), X11
	MOV	a_len+32(FP), X12
	MOV	b_base+48(FP), X14
	BEQZ	X12, bf16_vec_add_done
bf16_vec_add_loop:
	WORD	$0x0d0676d7           // vsetvli a3,a2,e32,m1,ta,ma
	WORD	$0x0205d007           // vle16.v v0,(a1)
	WORD	$0x02075087           // vle16.v v1,(a4)
	WORD	$0x4a032157           // vzext.vf2 v2,v0
	WORD	$0x4a132257           // vzext.vf2 v4,v1
	WORD	$0x96283157           // vsll.vi v2,v2,16
	WORD	$0x96483257           // vsll.vi v4,v4,16
	WORD	$0x02221357           // vfadd.vv v6,v2,v4
	WORD	$0xb2683457           // vnsrl.wi v8,v6,16
	WORD	$0x02055427           // vse16.v v8,(a0)
	SLLI	$1, X13, X15
	ADD	X15, X10, X10
	ADD	X15, X11, X11
	ADD	X15, X14, X14
	SUB	X13, X12, X12
	BNEZ	X12, bf16_vec_add_loop
bf16_vec_add_done:
	RET

// RVV encodings for BF16 dot/RMSNorm:
//   vle16.v v0,(a0)              0x02055007
//   vle16.v v1,(a1)              0x0205d087
//   vfmacc.vv v6,v2,v4           0xb2411357
//   vfredusum.vs v7,v6,v7        0x066393d7
//   vse32.v v7,(sp)              0x020163a7
//   vfmul.vv v6,v2,v4            0x92221357
//   vfmul.vf v6,v6,fa0           0x92655357

// func bf16DotAsm(x, y []uint16) float32
TEXT ·bf16DotAsm(SB), NOSPLIT, $16-52
	MOV	x_base+0(FP), X10
	MOV	x_len+8(FP), X12
	MOV	y_base+24(FP), X11
	BEQZ	X12, bf16_dot_zero

	MOV	$-1, X13
	WORD	$0x0906f057           // vsetvli zero,a3,e32,m1,tu,ma
	WORD	$0x5e003357           // vmv.v.i v6,0

bf16_dot_loop:
	WORD	$0x090676d7           // vsetvli a3,a2,e32,m1,tu,ma
	WORD	$0x02055007           // vle16.v v0,(a0)
	WORD	$0x0205d087           // vle16.v v1,(a1)
	WORD	$0x4a032157           // vzext.vf2 v2,v0
	WORD	$0x4a132257           // vzext.vf2 v4,v1
	WORD	$0x96283157           // vsll.vi v2,v2,16
	WORD	$0x96483257           // vsll.vi v4,v4,16
	WORD	$0xb2411357           // vfmacc.vv v6,v2,v4
	SLLI	$1, X13, X14
	ADD	X14, X10, X10
	ADD	X14, X11, X11
	SUB	X13, X12, X12
	BNEZ	X12, bf16_dot_loop

	MOV	$-1, X13
	WORD	$0x0906f057           // vsetvli zero,a3,e32,m1,tu,ma
	WORD	$0x5e0033d7           // vmv.v.i v7,0
	WORD	$0x066393d7           // vfredusum.vs v7,v6,v7
	MOV	$1, X13
	WORD	$0x0906f057           // vsetvli zero,a3,e32,m1,tu,ma
	WORD	$0x020163a7           // vse32.v v7,(sp)
	MOVF	0(SP), F0
	MOVF	F0, ret+48(FP)
	RET

bf16_dot_zero:
	MOV	$0, X15
	MOVW	X15, 0(SP)
	MOVF	0(SP), F0
	MOVF	F0, ret+48(FP)
	RET

// func bf16RMSNormScaleAsm(x, w []uint16, scale float32)
TEXT ·bf16RMSNormScaleAsm(SB), NOSPLIT, $0-52
	MOV	x_base+0(FP), X10
	MOV	x_len+8(FP), X12
	MOV	w_base+24(FP), X11
	MOVF	scale+48(FP), F10
	BEQZ	X12, bf16_rms_norm_done
bf16_rms_norm_loop:
	WORD	$0x0d0676d7           // vsetvli a3,a2,e32,m1,ta,ma
	WORD	$0x02055007           // vle16.v v0,(a0)
	WORD	$0x0205d087           // vle16.v v1,(a1)
	WORD	$0x4a032157           // vzext.vf2 v2,v0
	WORD	$0x4a132257           // vzext.vf2 v4,v1
	WORD	$0x96283157           // vsll.vi v2,v2,16
	WORD	$0x96483257           // vsll.vi v4,v4,16
	WORD	$0x92221357           // vfmul.vv v6,v2,v4
	WORD	$0x92655357           // vfmul.vf v6,v6,fa0
	WORD	$0xb2683457           // vnsrl.wi v8,v6,16
	WORD	$0x02055427           // vse16.v v8,(a0)
	SLLI	$1, X13, X14
	ADD	X14, X10, X10
	ADD	X14, X11, X11
	SUB	X13, X12, X12
	BNEZ	X12, bf16_rms_norm_loop
bf16_rms_norm_done:
	RET
