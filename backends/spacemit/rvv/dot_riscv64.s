#include "textflag.h"

// func dotI8(a, b *int8, n int64) int32
// RVV int8 dot product. Vector ops are WORD-encoded (Go asm has no RVV).
// Scalar regs used by the encoded vector insns: a0=X10 a1=X11 a2=X12 t0=X5.
TEXT ·dotI8(SB), NOSPLIT, $0-28
	MOV	a+0(FP), X10
	MOV	b+8(FP), X11
	MOV	n+16(FP), X12
	WORD	$0xcd00f357	// vsetivli t1,1,e32,m1,ta,ma
	WORD	$0x42006457	// vmv.s.x  v8,zero      (acc=0)
loop:
	BEQZ	X12, done
	WORD	$0x0c0672d7	// vsetvli  t0(X5),a2,e8,m1,ta,ma
	WORD	$0x02050087	// vle8.v   v1,(a0)
	WORD	$0x02058107	// vle8.v   v2,(a1)
	WORD	$0xee112257	// vwmul.vv v4,v1,v2
	WORD	$0x0c92f3d7	// vsetvli  t2,t0,e16,m2,ta,ma
	WORD	$0xc6440457	// vwredsum.vs v8,v4,v8
	ADD	X5, X10, X10	// a += vl
	ADD	X5, X11, X11	// b += vl
	SUB	X5, X12, X12	// n -= vl
	JMP	loop
done:
	WORD	$0xcd00f357	// vsetivli t1,1,e32,m1,ta,ma
	WORD	$0x42802557	// vmv.x.s  a0,v8
	MOVW	X10, ret+24(FP)
	RET
