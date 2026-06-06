#include "textflag.h"

// func kernelM4N32W4(a, b4 *int8, c *int32, K, lda, ldc int64)
// 4(M)x32(N) outer-product, uint8 A x int4 W -> int32. B4 per k = 16 bytes:
// byte j = (w[col j]&0xF) | ((w[col j+16]&0xF)<<4), 4-bit signed.
TEXT ·kernelM4N32W4(SB), NOSPLIT, $0-48
	MOV	a+0(FP), X10
	MOV	b4+8(FP), X11
	MOV	c+16(FP), X12
	MOV	K+24(FP), X13
	MOV	lda+32(FP), X14
	MOV	ldc+40(FP), X15
	ADD	X14, X10, X5
	ADD	X14, X5, X6
	ADD	X14, X6, X7
	MOV	$16, X31
	WORD	$0x0d1ff057	// vsetvli zero,t6,e32,m2
	WORD	$0x5e003857	// vmv.v.i v16,0
	WORD	$0x5e003957	// v18
	WORD	$0x5e003a57	// v20
	WORD	$0x5e003b57	// v22
	WORD	$0x5e003c57	// v24
	WORD	$0x5e003d57	// v26
	WORD	$0x5e003e57	// v28
	WORD	$0x5e003f57	// v30
loop:
	BEQZ	X13, store
	WORD	$0x0c0ff057	// vsetvli zero,t6,e8,m1
	WORD	$0x02058007	// vle8.v v0,(a1)
	ADD	$16, X11, X11
	WORD	$0x2607b0d7	// vand.vi v1,v0,15
	WORD	$0x2e1430d7	// vxor.vi v1,v1,8
	WORD	$0x021c30d7	// vadd.vi v1,v1,-8
	WORD	$0xa2023157	// vsrl.vi v2,v0,4
	WORD	$0x2e243157	// vxor.vi v2,v2,8
	WORD	$0x022c3157	// vadd.vi v2,v2,-8
	WORD	$0x0d1ff057	// vsetvli zero,t6,e32,m2
	WORD	$0x4a12a257	// vsext.vf4 v4,v1
	WORD	$0x4a22a457	// vsext.vf4 v8,v2
	MOVBU	(X10), X28
	ADD	$1, X10, X10
	WORD	$0xb64e6857	// vmacc.vx v16,t3,v4
	WORD	$0xb68e6c57	// vmacc.vx v24,t3,v8
	MOVBU	(X5), X29
	ADD	$1, X5, X5
	WORD	$0xb64ee957	// vmacc.vx v18,t4,v4
	WORD	$0xb68eed57	// vmacc.vx v26,t4,v8
	MOVBU	(X6), X30
	ADD	$1, X6, X6
	WORD	$0xb64f6a57	// vmacc.vx v20,t5,v4
	WORD	$0xb68f6e57	// vmacc.vx v28,t5,v8
	MOVBU	(X7), X28
	ADD	$1, X7, X7
	WORD	$0xb64e6b57	// vmacc.vx v22,t3,v4
	WORD	$0xb68e6f57	// vmacc.vx v30,t3,v8
	ADD	$-1, X13, X13
	JMP	loop
store:
	ADD	$64, X12, X29	// t4 = c + 16*int32
	WORD	$0x02066827	// vse32.v v16,(a2)
	WORD	$0x020eec27	// vse32.v v24,(t4)
	ADD	X15, X12, X12
	ADD	X15, X29, X29
	WORD	$0x02066927	// vse32.v v18,(a2)
	WORD	$0x020eed27	// vse32.v v26,(t4)
	ADD	X15, X12, X12
	ADD	X15, X29, X29
	WORD	$0x02066a27	// vse32.v v20,(a2)
	WORD	$0x020eee27	// vse32.v v28,(t4)
	ADD	X15, X12, X12
	ADD	X15, X29, X29
	WORD	$0x02066b27	// vse32.v v22,(a2)
	WORD	$0x020eef27	// vse32.v v30,(t4)
	RET
