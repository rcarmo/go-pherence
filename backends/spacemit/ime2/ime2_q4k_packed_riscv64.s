#include "textflag.h"
#include "ime2_isa.h"

// func vmadotQ4KPackedLoop(wQS *byte, actReord *byte, acc *int32, Kgroups int)
// Reads PACKED Q4K bytes (16 per 32 elements), unpacks with vand/vsrl,
// merges with vslideup, then vmadot against reordered activation.
// 2× less weight memory bandwidth than byte-per-nibble approach.
// Kgroups = K/32. acc = 64 bytes (int32[16], pre-loaded/stored).
TEXT ·vmadotQ4KPackedLoop(SB), NOSPLIT, $0-32
    MOV  wQS+0(FP), X10       // a0 = packed Q4K bytes (16 per group)
    MOV  actReord+8(FP), X11  // a1 = reordered act (32 bytes per group)
    MOV  acc+16(FP), X12      // a2 = accumulator
    MOV  Kgroups+24(FP), X13  // a3 = number of 32-element groups

    // Load accumulator
    WORD $0x011072d7            // vsetvli t0, zero, e32, m2
    WORD $0x02066e07            // vle32.v v28, (a2)

    WORD $0x000072d7            // vsetvli t0, zero, e8, m1 (vl=32)

pk_loop:
    BEQ  X13, X0, pk_done

    // Load 16 packed Q4K bytes into v0[0:15] (use vl=16)
    MOV  $16, X5
    WORD $0x0002f2d7            // vsetvli t0, t0, e8, m1 (vl=16)
    WORD $0x02050007            // vle8.v v0, (a0)

    // Extract: v1 = low nibbles, v2 = high nibbles
    WORD $0x2607b0d7            // vand.vi v1, v0, 15
    WORD $0xa2023157            // vsrl.vi v2, v0, 4

    // Merge into 32-byte weight tile: v4 = [v1[0:15], v2[0:15]]
    WORD $0x000072d7            // vsetvli t0, zero, e8, m1 (vl=32)
    WORD $0x5e008257            // vmv.v.v v4, v1          // v4[0:15] = low nibbles
    WORD $0x3a283257            // vslideup.vi v4, v2, 16  // v4[16:31] = high nibbles

    // Load 32-byte activation tile (reordered: even[16] + odd[16])
    WORD $0x02058187            // vle8.v v3, (a1)

    // v28 += v3 × v4^T : signed activation (vs1=v3) × unsigned 4-bit weight (vs2=v4)
    VMADOT_SU(28, 3, 4)

    // Advance pointers
    ADD  $16, X10, X10         // wQS += 16 bytes
    ADD  $32, X11, X11         // act += 32 bytes
    ADD  $-1, X13, X13         // Kgroups--
    JMP  pk_loop

pk_done:
    // Store accumulator
    WORD $0x011072d7            // vsetvli t0, zero, e32, m2
    WORD $0x02066e27            // vse32.v v28, (a2)
    RET
