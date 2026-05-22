//go:build riscv64

// sdot_riscv64.s — RISC-V RVV dot product kernel.
// Go's assembler currently exposes RVV opcodes but does not encode the vector
// operand forms we need, so vector instructions are emitted as raw WORDs from
// GNU as/objdump. Scalar control flow remains normal Go/Plan 9 assembly.

#include "textflag.h"

// RVV encodings used below:
//   vsetvli a3,a2,e32,m1,tu,ma   0x090676d7
//   vsetvli zero,a3,e32,m1,tu,ma 0x0906f057
//   vmv.v.i v2,0                 0x5e003157
//   vmv.v.i v3,0                 0x5e0031d7
//   vle32.v v0,(a0)              0x02056007
//   vle32.v v1,(a1)              0x0205e087
//   vfmacc.vv v2,v0,v1           0xb2101157
//   vfredusum.vs v3,v2,v3        0x062191d7
//   vse32.v v3,(sp)              0x020161a7

// func sdotAsm(x, y []float32) float32
TEXT ·sdotAsm(SB), NOSPLIT, $16-52
	MOV	x_base+0(FP), X10      // a0 = &x[0]
	MOV	x_len+8(FP), X12       // a2 = len(x)
	MOV	y_base+24(FP), X11     // a1 = &y[0]
	BEQZ	X12, zero

	// Set VL to VLMAX and clear the vector accumulator v2.
	MOV	$-1, X13              // a3 = AVL=-1 => VLMAX
	WORD	$0x0906f057           // vsetvli zero,a3,e32,m1,tu,ma
	WORD	$0x5e003157           // vmv.v.i v2,0

loop:
	WORD	$0x090676d7           // vsetvli a3,a2,e32,m1,tu,ma
	WORD	$0x02056007           // vle32.v v0,(a0)
	WORD	$0x0205e087           // vle32.v v1,(a1)
	WORD	$0xb2101157           // vfmacc.vv v2,v0,v1
	SLLI	$2, X13, X14          // byte count = vl * sizeof(float32)
	ADD	X14, X10, X10
	ADD	X14, X11, X11
	SUB	X13, X12, X12
	BNEZ	X12, loop

	// Reduce the full accumulator vector. The loop uses tail-undisturbed policy,
	// so lanes outside the final short VL retain earlier partial sums.
	MOV	$-1, X13
	WORD	$0x0906f057           // vsetvli zero,a3,e32,m1,tu,ma
	WORD	$0x5e0031d7           // vmv.v.i v3,0
	WORD	$0x062191d7           // vfredusum.vs v3,v2,v3

	// Store lane 0 only, then return it as float32.
	MOV	$1, X13
	WORD	$0x0906f057           // vsetvli zero,a3,e32,m1,tu,ma
	WORD	$0x020161a7           // vse32.v v3,(sp)
	MOVF	0(SP), F0
	MOVF	F0, ret+48(FP)
	RET

zero:
	MOV	$0, X15
	MOVW	X15, 0(SP)
	MOVF	0(SP), F0
	MOVF	F0, ret+48(FP)
	RET
