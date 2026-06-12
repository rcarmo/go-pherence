#include "textflag.h"
#include "ime2_isa.h"

// func vmadotKLoopAI(A *byte, B *byte, C *int32, K int)
// K-loop that forces vl=32 (VLEN=256 behavior) on AI cores (VLEN=1024).
// Uses vsetivli to cap vl at 32 regardless of hardware VLEN.
// Same 32-byte tiles as vmadotKLoop. Compatible with existing PackTiles format.
TEXT ·vmadotKLoopAI(SB), NOSPLIT, $0-32
    MOV  A+0(FP), X10
    MOV  B+8(FP), X11
    MOV  C+16(FP), X12
    MOV  K+24(FP), X13

    // Load accumulator (force e32, vl=16 for EMUL=2 int32 load)
    MOV  $16, X5
    WORD $0x0d12f2d7            // vsetivli t0 (actually vsetvli t0, t0, e32, m2, ta, ma... need correct enc)
    // Actually use vsetvli with explicit count in register:
    // set vl=16 for e32 (16 int32 = 64 bytes = accumulator)
    WORD $0x0112f2d7            // vsetvli t0, t0, e32, m2, tu, mu (with t0=16 from MOV above)
    WORD $0x02066e07            // vle32.v v28, (a2)

    // Force vl=32 for e8 operations (same as VLEN=256)
    MOV  $32, X5
    WORD $0x0002f2d7            // vsetvli t0, t0, e8, m1, tu, mu (vl=min(32, VLEN/8)=32)

    SRLI $3, X13, X14          // K/8 iterations (same as original)

loop_ai:
    BEQ  X14, X0, done_ai
    WORD $0x02050007            // vle8.v v0, (a0) — loads 32 bytes (vl=32)
    WORD $0x02058087            // vle8.v v1, (a1) — loads 32 bytes
    VMADOT_SS(28, 0, 1)         // vmadot v28, v1, v0
    ADD  $32, X10, X10
    ADD  $32, X11, X11
    ADD  $-1, X14, X14
    JMP  loop_ai

done_ai:
    // Store accumulator
    MOV  $16, X5
    WORD $0x0112f2d7            // vsetvli t0, t0, e32, m2
    WORD $0x02066e27            // vse32.v v28, (a2)
    RET

// func vmadotKLoop1024native(A *byte, B *byte, C *int32, K int)
// Full VLEN=1024 vmadot (vl=128, 128-byte tiles). No vl cap.
TEXT ·vmadotKLoop1024native(SB), NOSPLIT, $0-32
    MOV  A+0(FP), X10
    MOV  B+8(FP), X11
    MOV  C+16(FP), X12
    MOV  K+24(FP), X13
    // Load acc (e32, m2 → 64 int32s on VLEN=1024)
    WORD $0x011072d7            // vsetvli t0, zero, e32, m2
    WORD $0x02066e07            // vle32.v v28, (a2)
    // Set e8, m1 (vl=128 on VLEN=1024)
    WORD $0x000072d7            // vsetvli t0, zero, e8, m1
    // K/16 iterations (128 bytes per tile = 16 elements per row × 8 rows)
    SRLI $4, X13, X14
native_loop:
    BEQ  X14, X0, native_done
    WORD $0x02050007            // vle8.v v0, (a0)
    WORD $0x02058087            // vle8.v v1, (a1)
    VMADOT_SS(28, 0, 1)         // vmadot v28, v1, v0
    ADD  $128, X10, X10
    ADD  $128, X11, X11
    ADD  $-1, X14, X14
    JMP  native_loop
native_done:
    WORD $0x011072d7            // vsetvli t0, zero, e32, m2
    WORD $0x02066e27            // vse32.v v28, (a2)
    RET
