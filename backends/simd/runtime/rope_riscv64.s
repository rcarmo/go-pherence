//go:build riscv64

// rope_riscv64.s — RVV partial RoPE head kernel.

#include "textflag.h"

// Encodings from GNU as/objdump:
//   vsetvli a3,a2,e32,m1,ta,ma  0x0d0676d7
//   vle32.v v0,(a0)             0x02056007
//   vle32.v v1,(a1)             0x0205e087
//   vlse32.v v2,(a4),t0         0x0a576107
//   vlse32.v v3,(a5),t0         0x0a57e187
//   vfmul.vv v4,v0,v2           0x92011257
//   vfmul.vv v6,v1,v3           0x92119357
//   vfsub.vv v4,v4,v6           0x0a431257
//   vfmul.vv v5,v0,v3           0x920192d7
//   vfmacc.vv v5,v1,v2          0xb22092d7
//   vse32.v v4,(a0)             0x02056227
//   vse32.v v5,(a1)             0x0205e2a7

// func ropePartialHeadAsm(x0, x1, freqs []float32, n int)
TEXT ·ropePartialHeadAsm(SB), NOSPLIT, $0-80
	MOV	x0_base+0(FP), X10
	MOV	x1_base+24(FP), X11
	MOV	freqs_base+48(FP), X14
	MOV	n+72(FP), X12
	BEQZ	X12, rope_done
	MOV	$8, X5                 // stride between cos/sin entries
	ADD	$4, X14, X15           // sin starts one float after cos
rope_loop:
	WORD	$0x0d0676d7           // vsetvli a3,a2,e32,m1,ta,ma
	WORD	$0x02056007           // vle32.v v0,(a0)      x0
	WORD	$0x0205e087           // vle32.v v1,(a1)      x1
	WORD	$0x0a576107           // vlse32.v v2,(a4),t0 cos
	WORD	$0x0a57e187           // vlse32.v v3,(a5),t0 sin
	WORD	$0x92011257           // vfmul.vv v4,v0,v2   x0*cos
	WORD	$0x92119357           // vfmul.vv v6,v1,v3   x1*sin
	WORD	$0x0a431257           // vfsub.vv v4,v4,v6   out0
	WORD	$0x920192d7           // vfmul.vv v5,v0,v3   x0*sin
	WORD	$0xb22092d7           // vfmacc.vv v5,v1,v2  +x1*cos
	WORD	$0x02056227           // vse32.v v4,(a0)
	WORD	$0x0205e2a7           // vse32.v v5,(a1)
	SLLI	$2, X13, X16
	ADD	X16, X10, X10
	ADD	X16, X11, X11
	SLLI	$3, X13, X16
	ADD	X16, X14, X14
	ADD	X16, X15, X15
	SUB	X13, X12, X12
	BNEZ	X12, rope_loop
rope_done:
	RET
