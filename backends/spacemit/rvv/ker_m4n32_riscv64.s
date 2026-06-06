#include "textflag.h"

// func kernelM4N32(a, bp *int8, c *int32, K, lda, ldc int64)
// 4(M) x 32(N) outer-product int8->int32 tile. Bp is [K][32] packed.
// Vector ops WORD-encoded; scalar regs match: a0=X10 a1=X11 a2=X12 a3=X13
// a4=X14 a5=X15 t0=X5 t1=X6 t2=X7 t3=X28 t4=X29 t5=X30 t6=X31.
TEXT ·kernelM4N32(SB), NOSPLIT, $0-48
	MOV	a+0(FP), X10
	MOV	bp+8(FP), X11
	MOV	c+16(FP), X12
	MOV	K+24(FP), X13
	MOV	lda+32(FP), X14
	MOV	ldc+40(FP), X15
	ADD	X14, X10, X5	// t0 = a + lda
	ADD	X14, X5, X6	// t1
	ADD	X14, X6, X7	// t2
	MOV	$32, X31
	WORD	$0x0d2ff057	// vsetvli zero,t6,e32,m4,ta,ma
	WORD	$0x5e003457	// vmv.v.i v8,0
	WORD	$0x5e003657	// vmv.v.i v12,0
	WORD	$0x5e003857	// vmv.v.i v16,0
	WORD	$0x5e003a57	// vmv.v.i v20,0
loop:
	BEQZ	X13, store
	WORD	$0x02058007	// vle8.v v0,(a1)
	ADD	$32, X11, X11	// bp += 32
	WORD	$0x4a02a257	// vsext.vf4 v4,v0
	MOVB	(X10), X28	// lb t3,(a0)
	ADD	$1, X10, X10
	WORD	$0xb64e6457	// vmacc.vx v8,t3,v4
	MOVB	(X5), X29	// lb t4,(t0)
	ADD	$1, X5, X5
	WORD	$0xb64ee657	// vmacc.vx v12,t4,v4
	MOVB	(X6), X30	// lb t5,(t1)
	ADD	$1, X6, X6
	WORD	$0xb64f6857	// vmacc.vx v16,t5,v4
	MOVB	(X7), X28	// lb t3,(t2)
	ADD	$1, X7, X7
	WORD	$0xb64e6a57	// vmacc.vx v20,t3,v4
	ADD	$-1, X13, X13
	JMP	loop
store:
	WORD	$0x02066427	// vse32.v v8,(a2)
	ADD	X15, X12, X12
	WORD	$0x02066627	// vse32.v v12,(a2)
	ADD	X15, X12, X12
	WORD	$0x02066827	// vse32.v v16,(a2)
	ADD	X15, X12, X12
	WORD	$0x02066a27	// vse32.v v20,(a2)
	RET
