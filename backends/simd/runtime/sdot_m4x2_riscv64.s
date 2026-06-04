//go:build riscv64

// sdot_m4x2_riscv64.s — RISC-V RVV dot product, LMUL=m4 with two alternating
// accumulators to hide vector-FMA latency on the in-order K1 pipe.
//
// sdotM4Asm accumulates every chunk into one register group, so each vfmacc
// depends on the previous one and the loop runs at FMA *latency*, not
// throughput. This kernel alternates consecutive chunks between two independent
// accumulator groups (v8 and v16), breaking the dependency chain so the pipe can
// stay busy. The two partial sums are added before the final reduction.
//
// RVV encodings (e32,m4,tu,ma):
//   vsetvli a3,a2,e32,m4,tu,ma     0x092676d7
//   vsetvli zero,a3,e32,m4,tu,ma   0x0926f057
//   vmv.v.i v8,0                   0x5e003457
//   vmv.v.i v16,0                  0x5e003857
//   vmv.v.i v24,0                  0x5e003c57
//   vle32.v v0,(a0)                0x02056007
//   vle32.v v4,(a1)                0x0205e207
//   vfmacc.vv v8,v0,v4             0xb2401457
//   vfmacc.vv v16,v0,v4            0xb2401857
//   vfadd.vv v8,v16,v8             0x03041457
//   vfredusum.vs v24,v8,v24        0x068c1c57
//   vse32.v v24,(t0)               0x0202ec27

#include "textflag.h"

// func sdotM4x2Asm(x, y []float32) float32
TEXT ·sdotM4x2Asm(SB), NOSPLIT, $16-52
	MOV	x_base+0(FP), X10
	MOV	x_len+8(FP), X12
	MOV	y_base+24(FP), X11
	BEQZ	X12, zero

	MOV	$-1, X13
	WORD	$0x0926f057           // vsetvli zero,a3,e32,m4 (VLMAX)
	WORD	$0x5e003457           // vmv.v.i v8,0
	WORD	$0x5e003857           // vmv.v.i v16,0

loop:
	// accumulator 0
	WORD	$0x092676d7           // vsetvli a3,a2,e32,m4
	WORD	$0x02056007           // vle32.v v0,(a0)
	WORD	$0x0205e207           // vle32.v v4,(a1)
	WORD	$0xb2401457           // vfmacc.vv v8,v0,v4
	SLLI	$2, X13, X14
	ADD	X14, X10, X10
	ADD	X14, X11, X11
	SUB	X13, X12, X12
	BEQZ	X12, reduce

	// accumulator 1
	WORD	$0x092676d7           // vsetvli a3,a2,e32,m4
	WORD	$0x02056007           // vle32.v v0,(a0)
	WORD	$0x0205e207           // vle32.v v4,(a1)
	WORD	$0xb2401857           // vfmacc.vv v16,v0,v4
	SLLI	$2, X13, X14
	ADD	X14, X10, X10
	ADD	X14, X11, X11
	SUB	X13, X12, X12
	BNEZ	X12, loop

reduce:
	MOV	$-1, X13
	WORD	$0x0926f057           // vsetvli zero,a3 (VLMAX)
	WORD	$0x03041457           // vfadd.vv v8,v16,v8
	WORD	$0x5e003c57           // vmv.v.i v24,0
	WORD	$0x068c1c57           // vfredusum.vs v24,v8,v24

	MOV	$1, X13
	WORD	$0x0926f057           // vsetvli zero,a3 (VL=1)
	ADDI	$8, X2, X5
	WORD	$0x0202ec27           // vse32.v v24,(t0)
	MOVF	8(SP), F0
	MOVF	F0, ret+48(FP)
	RET

zero:
	MOV	$0, X15
	MOVW	X15, 8(SP)
	MOVF	8(SP), F0
	MOVF	F0, ret+48(FP)
	RET
