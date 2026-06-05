//go:build riscv64

// sdot_m8_riscv64.s — RISC-V RVV dot product at LMUL=m8 (max group).
//
// Pushes the LMUL-widening idea to the maximum 8-register group (VLMAX =
// 8*VLEN/32). Fewest possible loop iterations / dependent vfmacc ops for the
// in-order K1 vector pipe. Uses register groups v0..v7 (a), v8..v15 (b),
// v16..v23 (accumulator), v24 (reduction).
//
// RVV encodings (e32,m8,tu,ma):
//   vsetvli a3,a2,e32,m8,tu,ma     0x093676d7
//   vsetvli zero,a3,e32,m8,tu,ma   0x0936f057
//   vmv.v.i v16,0                  0x5e003857
//   vmv.v.i v24,0                  0x5e003c57
//   vle32.v v0,(a0)                0x02056007   (group v0..v7)
//   vle32.v v8,(a1)                0x0205e407   (group v8..v15)
//   vfmacc.vv v16,v0,v8            0xb2801857   (v16grp += v0grp * v8grp)
//   vfredusum.vs v24,v16,v24       0x070c1c57
//   vse32.v v24,(t0)               0x0202ec27

#include "textflag.h"

// func sdotM8Asm(x, y []float32) float32
TEXT ·sdotM8Asm(SB), NOSPLIT, $16-52
	MOV	x_base+0(FP), X10
	MOV	x_len+8(FP), X12
	MOV	y_base+24(FP), X11
	BEQZ	X12, zero

	MOV	$-1, X13
	WORD	$0x0936f057           // vsetvli zero,a3,e32,m8,tu,ma (VLMAX)
	WORD	$0x5e003857           // vmv.v.i v16,0

loop:
	WORD	$0x093676d7           // vsetvli a3,a2,e32,m8,tu,ma
	WORD	$0x02056007           // vle32.v v0,(a0)
	WORD	$0x0205e407           // vle32.v v8,(a1)
	WORD	$0xb2801857           // vfmacc.vv v16,v0,v8
	SLLI	$2, X13, X14
	ADD	X14, X10, X10
	ADD	X14, X11, X11
	SUB	X13, X12, X12
	BNEZ	X12, loop

	MOV	$-1, X13
	WORD	$0x0936f057           // vsetvli zero,a3 (VLMAX)
	WORD	$0x5e003c57           // vmv.v.i v24,0
	WORD	$0x070c1c57           // vfredusum.vs v24,v16,v24

	MOV	$1, X13
	WORD	$0x0936f057           // vsetvli zero,a3 (VL=1)
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
