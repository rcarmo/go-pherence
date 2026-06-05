//go:build riscv64

// sgemm_riscv64.s — RISC-V RVV register-blocked SGEMM microkernel.
//
// Computes four dot products at once: one A row against four contiguous B rows
// (the NT / A*B^T layout, where each row is contiguous in memory). The A vector
// chunk is loaded once and reused across all four B rows, and four vector
// accumulators stay live in registers across the whole k loop. Only four
// horizontal reductions happen at the very end (vs one per output cell in the
// scalar-orchestrated path). Vector ops are emitted as raw WORDs, matching the
// convention in sdot_riscv64.s.
//
// RVV encodings used:
//   vsetvli a3,a2,e32,m1,tu,ma     0x090676d7
//   vsetvli zero,a3,e32,m1,tu,ma   0x0906f057
//   vmv.v.i v8,0                   0x5e003457
//   vmv.v.i v9,0                   0x5e0034d7
//   vmv.v.i v10,0                  0x5e003557
//   vmv.v.i v11,0                  0x5e0035d7
//   vmv.v.i v12,0                  0x5e003657
//   vle32.v v0,(a0=X10)            0x02056007
//   vle32.v v1,(X11)              0x0205e087
//   vle32.v v2,(X16)              0x02086107
//   vle32.v v3,(X17)              0x0208e187
//   vle32.v v4,(X18)              0x02096207
//   vfmacc.vv v8,v0,v1            0xb2101457
//   vfmacc.vv v9,v0,v2            0xb22014d7
//   vfmacc.vv v10,v0,v3           0xb2301557
//   vfmacc.vv v11,v0,v4           0xb24015d7
//   vfredusum.vs v12,v8,v12       0x06861657
//   vfredusum.vs v12,v9,v12       0x06961657
//   vfredusum.vs v12,v10,v12      0x06a61657
//   vfredusum.vs v12,v11,v12      0x06b61657
//   vse32.v v12,(t0=X5)           0x0202e627

#include "textflag.h"

// func sgemmNT1x4Asm(a, b *float32, ldb int, k int, sums *float32)
//   a    : pointer to one A row (k contiguous float32)
//   b    : pointer to first of four B rows (each k contiguous float32)
//   ldb  : byte stride between consecutive B rows
//   k    : dot length
//   sums : pointer to 4 float32 outputs (raw dot products, no alpha/bias)
TEXT ·sgemmNT1x4Asm(SB), NOSPLIT, $16-40
	MOV	a+0(FP), X10          // a0 = &a[0]
	MOV	b+8(FP), X11          // b0
	MOV	ldb+16(FP), X14       // byte stride between B rows
	MOV	k+24(FP), X12         // a2 = remaining k (AVL)
	MOV	sums+32(FP), X19      // &sums[0]

	// b1, b2, b3 = b0 + ldb, +2*ldb, +3*ldb
	ADD	X14, X11, X16
	ADD	X14, X16, X17
	ADD	X14, X17, X18

	// Zero the four accumulators at VLMAX.
	MOV	$-1, X13
	WORD	$0x0906f057           // vsetvli zero,a3,e32,m1,tu,ma
	WORD	$0x5e003457           // vmv.v.i v8,0
	WORD	$0x5e0034d7           // vmv.v.i v9,0
	WORD	$0x5e003557           // vmv.v.i v10,0
	WORD	$0x5e0035d7           // vmv.v.i v11,0

	BEQZ	X12, reduce

loop:
	WORD	$0x090676d7           // vsetvli a3,a2,e32,m1,tu,ma
	WORD	$0x02056007           // vle32.v v0,(a0)
	WORD	$0x0205e087           // vle32.v v1,(X11)
	WORD	$0x02086107           // vle32.v v2,(X16)
	WORD	$0x0208e187           // vle32.v v3,(X17)
	WORD	$0x02096207           // vle32.v v4,(X18)
	WORD	$0xb2101457           // vfmacc.vv v8,v0,v1
	WORD	$0xb22014d7           // vfmacc.vv v9,v0,v2
	WORD	$0xb2301557           // vfmacc.vv v10,v0,v3
	WORD	$0xb24015d7           // vfmacc.vv v11,v0,v4
	SLLI	$2, X13, X15          // byte advance = vl * 4
	ADD	X15, X10, X10
	ADD	X15, X11, X11
	ADD	X15, X16, X16
	ADD	X15, X17, X17
	ADD	X15, X18, X18
	SUB	X13, X12, X12
	BNEZ	X12, loop

reduce:
	// sums[0] = reduce(v8)
	MOV	$-1, X13
	WORD	$0x0906f057           // vsetvli zero,a3 (VLMAX)
	WORD	$0x5e003657           // vmv.v.i v12,0
	WORD	$0x06861657           // vfredusum.vs v12,v8,v12
	MOV	$1, X13
	WORD	$0x0906f057           // vsetvli zero,a3 (VL=1)
	ADDI	$8, X2, X5
	WORD	$0x0202e627           // vse32.v v12,(t0)
	MOVF	8(SP), F0
	MOVF	F0, 0(X19)

	// sums[1] = reduce(v9)
	MOV	$-1, X13
	WORD	$0x0906f057
	WORD	$0x5e003657           // vmv.v.i v12,0
	WORD	$0x06961657           // vfredusum.vs v12,v9,v12
	MOV	$1, X13
	WORD	$0x0906f057
	ADDI	$8, X2, X5
	WORD	$0x0202e627
	MOVF	8(SP), F0
	MOVF	F0, 4(X19)

	// sums[2] = reduce(v10)
	MOV	$-1, X13
	WORD	$0x0906f057
	WORD	$0x5e003657
	WORD	$0x06a61657           // vfredusum.vs v12,v10,v12
	MOV	$1, X13
	WORD	$0x0906f057
	ADDI	$8, X2, X5
	WORD	$0x0202e627
	MOVF	8(SP), F0
	MOVF	F0, 8(X19)

	// sums[3] = reduce(v11)
	MOV	$-1, X13
	WORD	$0x0906f057
	WORD	$0x5e003657
	WORD	$0x06b61657           // vfredusum.vs v12,v11,v12
	MOV	$1, X13
	WORD	$0x0906f057
	ADDI	$8, X2, X5
	WORD	$0x0202e627
	MOVF	8(SP), F0
	MOVF	F0, 12(X19)

	RET
