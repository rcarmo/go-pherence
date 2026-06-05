//go:build riscv64

// sdot_m4_riscv64.s — RISC-V RVV dot product at LMUL=m4.
//
// The baseline sdotAsm uses e32,m1 (one vector register, VLMAX=VLEN/32 f32) with
// a single accumulator. On the in-order, single-issue SpaceMIT K1 vector pipe
// that path is latency/overhead-bound (~4.3 GFLOP/s, ~17% of FMA peak): every
// loop iteration pays vsetvli + branch overhead, and each vfmacc depends on the
// previous one. Widening to m4 (a 4-register group, VLMAX = 4*VLEN/32) does 4x
// the work per instruction, so the k loop runs 4x fewer iterations and the
// dependent vfmacc chain is 4x shorter — directly attacking the overhead/latency
// wall. Vector ops are raw WORDs, same convention as sdot_riscv64.s.
//
// RVV encodings (e32,m4,tu,ma):
//   vsetvli a3,a2,e32,m4,tu,ma     0x092676d7
//   vsetvli zero,a3,e32,m4,tu,ma   0x0926f057
//   vmv.v.i v8,0                   0x5e003457   (zeros group v8..v11)
//   vmv.v.i v12,0                  0x5e003657
//   vle32.v v0,(a0)                0x02056007   (loads group v0..v3)
//   vle32.v v4,(a1)                0x0205e207   (loads group v4..v7)
//   vfmacc.vv v8,v0,v4             0xb2401457   (v8group += v0group * v4group)
//   vfredusum.vs v12,v8,v12        0x06861657
//   vse32.v v12,(t0)               0x0202e627

#include "textflag.h"

// func sdotM4Asm(x, y []float32) float32
TEXT ·sdotM4Asm(SB), NOSPLIT, $16-52
	MOV	x_base+0(FP), X10     // a0 = &x[0]
	MOV	x_len+8(FP), X12      // a2 = len(x) = AVL
	MOV	y_base+24(FP), X11    // a1 = &y[0]
	BEQZ	X12, zero

	MOV	$-1, X13
	WORD	$0x0926f057           // vsetvli zero,a3,e32,m4,tu,ma  (VLMAX)
	WORD	$0x5e003457           // vmv.v.i v8,0

loop:
	WORD	$0x092676d7           // vsetvli a3,a2,e32,m4,tu,ma
	WORD	$0x02056007           // vle32.v v0,(a0)   -> v0..v3
	WORD	$0x0205e207           // vle32.v v4,(a1)   -> v4..v7
	WORD	$0xb2401457           // vfmacc.vv v8,v0,v4
	SLLI	$2, X13, X14          // bytes = vl * 4
	ADD	X14, X10, X10
	ADD	X14, X11, X11
	SUB	X13, X12, X12
	BNEZ	X12, loop

	// Reduce the m4 accumulator group (VLMAX lanes) to a scalar.
	MOV	$-1, X13
	WORD	$0x0926f057           // vsetvli zero,a3 (VLMAX)
	WORD	$0x5e003657           // vmv.v.i v12,0
	WORD	$0x06861657           // vfredusum.vs v12,v8,v12

	MOV	$1, X13
	WORD	$0x0926f057           // vsetvli zero,a3 (VL=1)
	ADDI	$8, X2, X5
	WORD	$0x0202e627           // vse32.v v12,(t0)
	MOVF	8(SP), F0
	MOVF	F0, ret+48(FP)
	RET

zero:
	MOV	$0, X15
	MOVW	X15, 8(SP)
	MOVF	8(SP), F0
	MOVF	F0, ret+48(FP)
	RET
