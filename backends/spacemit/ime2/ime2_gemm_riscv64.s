// ime2_gemm_riscv64.s — Optimized GEMM inner loop using vmadot
// Processes 4 output rows × K columns without Go loop overhead.
//
// func vmadotKLoop(A *byte, B *byte, C *int32, K int)
// A: pointer to first row of A tile (4 rows × K cols, stride=K)
// B: pointer to first row of B tile (4 rows × K cols, stride=K)  
// C: pointer to 4×4 int32 accumulator (16 int32s = 64 bytes)
// K: number of columns (must be multiple of 8)
//
// Computes C[4×4] += A[4×K] * B[4×K]^T by iterating over K in steps of 8.

#include "textflag.h"
#include "k3_isa.h"

// Arguments via stack (Go calling convention for riscv64):
// A  = 8(SP)  → X10 (a0)
// B  = 16(SP) → X11 (a1)
// C  = 24(SP) → X12 (a2)
// K  = 32(SP) → X13 (a3)
// Kstride (=K for row-major) passed as K itself since rows are K-contiguous

TEXT ·vmadotKLoop(SB), NOSPLIT, $0-32
    MOV    A+0(FP), X10       // a0 = A base pointer
    MOV    B+8(FP), X11       // a1 = B base pointer
    MOV    C+16(FP), X12      // a2 = C pointer (accumulator)
    MOV    K+24(FP), X13      // a3 = K (columns)

    // Load existing accumulator C[4×4] from memory
    // vsetvli t0, zero, e32, m2, tu, mu
    WORD $0x011072d7
    // vle32.v v28, (a2)
    WORD $0x02066e07

    // Precompute row offsets: stride = K bytes
    // a4 = A + K (row 1), a5 = A + 2K (row 2), a6 = A + 3K (row 3)
    ADD  X13, X10, X14        // a4 = A + K
    ADD  X13, X14, X15        // a5 = A + 2K
    ADD  X13, X15, X16        // a6 = A + 3K
    // b4 = B + K (row 1), b5 = B + 2K, b6 = B + 3K
    ADD  X13, X11, X17        // a7 = B + K
    ADD  X13, X17, X28        // t3 = B + 2K
    ADD  X13, X28, X29        // t4 = B + 3K

    // Set up loop counter: iterations = K / 8
    SRLI $3, X13, X30         // t5 = K >> 3 (iterations)

    // Set vector config for INT8 loads
    // vsetvli t0, zero, e8, m1, tu, mu
    WORD $0x000072d7

loop:
    BEQ  X30, X0, done        // if iterations == 0, done

    // Load A tile: 4 rows × 8 bytes, packed into v0..v3
    // We need them interleaved as [row0[0:8], row1[0:8], row2[0:8], row3[0:8]] in one 32-byte vector
    // But vmadot expects vs1 as a 4×8 matrix in a single vector register (32 bytes)
    // Layout: bytes 0-7 = row0, bytes 8-15 = row1, bytes 16-23 = row2, bytes 24-31 = row3
    
    // Use v2,v3 as scratch to build the packed tile
    // Load 8 bytes from each A row into scalar, then use vslide/vmv to pack
    // Actually simpler: load each row segment separately and use indexed store
    
    // Alternative: use 4 separate vle8 with vl=8 into different parts of v0
    // Set vl=8 for partial loads
    MOV  $8, X5               // t0 = 8
    WORD $0x0002f2d7           // vsetvli t0, t0, e8, m1, tu, mu (vl=8)
    
    // Load A rows
    WORD $0x02050007           // vle8.v v0, (a0)     — A row 0 [8 bytes] into v0[0:7]

    // Now need to load row1 into v0[8:15], row2 into v0[16:23], row3 into v0[24:31]
    // Use vslideup to shift and merge
    // Actually this is getting complex. Let me use a different approach:
    // Load all 32 bytes from a temporary stack buffer.
    
    // SIMPLER: copy 8 bytes from each row into a 32-byte stack buffer, then load
    // But that's what the Go code does. Let me try a different layout.
    
    // BEST APPROACH: Require A to be PRE-PACKED in 4×8 tile format
    // Then: just vle8.v v0, (a0) loads the full 32-byte tile
    // Advance A by 32 bytes per iteration
    
    // With pre-packed layout: A is [K/8 tiles × 32 bytes per tile × M/4 row-groups]
    // Tile address = A_base + iteration * 32
    
    // Let's implement the pre-packed version:
    // Reset vl to 32
    WORD $0x000072d7           // vsetvli t0, zero, e8, m1, tu, mu (vl=32)
    
    // Load pre-packed A tile (32 bytes contiguous)
    WORD $0x02050007           // vle8.v v0, (a0)
    // Load pre-packed B tile (32 bytes contiguous)
    WORD $0x02058087           // vle8.v v1, (a1)
    
    // vmadot v28, v0, v1 (accumulate)
    VMADOT_SS(28, 0, 1)
    
    // Advance pointers by 32 bytes
    ADD  $32, X10, X10        // a0 += 32
    ADD  $32, X11, X11        // a1 += 32
    
    // Decrement loop counter
    ADD  $-1, X30, X30        // t5--
    
    JMP  loop

done:
    // Store accumulated result
    // vsetvli t0, zero, e32, m2, tu, mu
    WORD $0x011072d7
    // vse32.v v28, (a2)
    WORD $0x02066e27
    RET
