// ime2_riscv64.s — SpacemiT IME2 vmadot instructions for RISC-V 64
// Build tag: riscv64
//
// vmadot encoding: (funct7<<26) | (1<<25) | (vs2<<20) | (vs1<<15) | (funct3<<12) | (vd<<7) | 0x2b
// funct7=0x38 (111000), funct3: 0=uu, 1=us, 2=su, 3=ss

#include "textflag.h"
#include "ime2_isa.h"

// func vmadotSS4x8(A *byte, B *byte, C *int32)
// Computes C[4x4] += A[4x8] * B[4x8]^T using vmadot (signed x signed)
// A, B are 32 bytes each (4 rows x 8 cols of int8)
// C is 32 bytes (4x4 int32, stored in 2 vector registers)
TEXT ·vmadotSS4x8(SB), NOSPLIT, $0-24
    MOV    A+0(FP), X10     // a0 = A pointer
    MOV    B+8(FP), X11     // a1 = B pointer
    MOV    C+16(FP), X12    // a2 = C pointer

    // vsetvli t0, zero, e8, m1, tu, mu
    WORD $0x000072d7

    // vle8.v v0, (a0) — load A[32 bytes]
    WORD $0x02050007

    // vle8.v v1, (a1) — load B[32 bytes]  
    WORD $0x02058087

    // vxor.vv v28, v28, v28 — zero accumulator
    WORD $0x2fce0e57

    // vxor.vv v29, v29, v29 — zero accumulator (EMUL=2)
    WORD $0x2fee8ed7

    // vmadot v28, v0, v1 (signed x signed, funct3=3)
    // encoding: (0x38<<26)|(1<<25)|(1<<20)|(0<<15)|(3<<12)|(28<<7)|0x2b = 0xe2103e2b
    VMADOT_SS(28, 0, 1)

    // vsetvli t0, zero, e32, m2, tu, mu
    WORD $0x011072d7

    // vse32.v v28, (a2) — store C[32 bytes]
    WORD $0x02066e27

    RET

// func vmadotUS4x8(A *byte, B *byte, C *int32)
// unsigned A x signed B
TEXT ·vmadotUS4x8(SB), NOSPLIT, $0-24
    MOV    A+0(FP), X10
    MOV    B+8(FP), X11
    MOV    C+16(FP), X12

    WORD $0x000072d7        // vsetvli t0, zero, e8, m1
    WORD $0x02050007        // vle8.v v0, (a0)
    WORD $0x02058087        // vle8.v v1, (a1)
    WORD $0x2fce0e57        // vxor.vv v28, v28, v28
    WORD $0x2fee8ed7        // vxor.vv v29, v29, v29
    // vmadotus: funct3=1
    // (0x38<<26)|(1<<25)|(1<<20)|(0<<15)|(1<<12)|(28<<7)|0x2b = 0xe2101e2b
    VMADOT_SU(28, 0, 1)
    WORD $0x011072d7        // vsetvli t0, zero, e32, m2
    WORD $0x02066e27        // vse32.v v28, (a2)
    RET

// func vmadotAccSS4x8(A *byte, B *byte, C *int32)
// Accumulating version: loads existing C, adds result
TEXT ·vmadotAccSS4x8(SB), NOSPLIT, $0-24
    MOV    A+0(FP), X10
    MOV    B+8(FP), X11
    MOV    C+16(FP), X12

    // Load existing accumulator from C
    WORD $0x011072d7        // vsetvli t0, zero, e32, m2
    WORD $0x02066e07        // vle32.v v28, (a2)

    // Set up for int8 loads
    WORD $0x000072d7        // vsetvli t0, zero, e8, m1
    WORD $0x02050007        // vle8.v v0, (a0)
    WORD $0x02058087        // vle8.v v1, (a1)

    // vmadot accumulates into v28 (adds to existing)
    VMADOT_SS(28, 0, 1)         // vmadot v28, v0, v1

    // Store result
    WORD $0x011072d7        // vsetvli t0, zero, e32, m2
    WORD $0x02066e27        // vse32.v v28, (a2)
    RET
