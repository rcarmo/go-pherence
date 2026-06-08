#include "textflag.h"
#include "k3_isa.h"

// func vmadotQ4KIntLoop1024x4(wBase *byte, wRGStride int, actBcast *byte, scratch *int32, intBuf *int32, intRGStride int, numSubs int)
//
// Processes 4 consecutive row groups per call, loading actBcast ONCE per tile
// instead of once per (row-group × tile). Reduces actBcast DRAM reads 4×.
//
// For each subblock sb = 0..numSubs-1:
//   1. Zero all 4 accumulators v28/v26/v24/v22 via vmv.v.i.
//   2. Per tile (2 tiles per subblock = 32 columns total):
//      - Load activation v1 from actBcast (SHARED by all 4 row groups).
//      - Load weights v0/v2/v4/v6 for rg0..rg3 and run vmadot.
//   3. Extract 8 diagonal int32s from each accumulator into intBuf.
//
// Register map (all caller-saved):
//   X10 (a0) = wBase rg0       (advances 128/tile)
//   X11 (a1) = wRGStride       (bytes between row groups in wPacked; constant)
//   X12 (a2) = actBcast        (advances 128/tile; shared)
//   X13 (a3) = scratch         (static 256-byte buffer)
//   X14 (a4) = intBuf rg0      (advances 32/subblock = 8 int32)
//   X15 (a5) = intRGStride     (bytes between row groups in intBuf; constant)
//   X16 (a6) = numSubs countdown
//   X5       = wBase rg1       (computed at entry)
//   X6       = wBase rg2
//   X7       = wBase rg3
//   X17      = intBuf rg1
//   X28      = intBuf rg2
//   X29      = intBuf rg3
//   X30      = LW/SW extraction temp

TEXT ·vmadotQ4KIntLoop1024x4(SB), NOSPLIT, $0-56
    MOV wBase+0(FP),       X10
    MOV wRGStride+8(FP),   X11
    MOV actBcast+16(FP),   X12
    MOV scratch+24(FP),    X13
    MOV intBuf+32(FP),     X14
    MOV intRGStride+40(FP), X15
    MOV numSubs+48(FP),    X16

    BEQ X16, X0, done_x4

    // Compute derived pointers
    WORD $0x00B502B3    // ADD X5, X10, X11  → wBase rg1 = wBase + wRGStride
    WORD $0x00B28333    // ADD X6, X5, X11   → wBase rg2
    WORD $0x00B303B3    // ADD X7, X6, X11   → wBase rg3
    WORD $0x00F708B3    // ADD X17, X14, X15 → intBuf rg1
    WORD $0x00F88E33    // ADD X28, X17, X15 → intBuf rg2
    WORD $0x00FE0EB3    // ADD X29, X28, X15 → intBuf rg3

    // Set e32,m2 once (for vmv.v.i to zero 64-element accumulators)
    WORD $0x01107FD7        // vsetvli t0, zero, e32, m2

subloop_x4:
    // === 1. Zero all 4 accumulators (e32,m2 → 64 int32 each) ===
    WORD $0x5E003E57        // vmv.v.i v28, 0  (rg0 acc)
    WORD $0x5E003D57        // vmv.v.i v26, 0  (rg1 acc)
    WORD $0x5E003C57        // vmv.v.i v24, 0  (rg2 acc)
    WORD $0x5E003B57        // vmv.v.i v22, 0  (rg3 acc)

    // === 2. Two vmadot passes (e8,m1 → vl=128) ===
    WORD $0x00007FD7        // vsetvli t0, zero, e8, m1

    // Tile 1 (columns 0-15):
    WORD $0x02060087        // vle8.v v1, (X12=actBcast)  -- shared activation
    WORD $0x02050007        // vle8.v v0, (X10=wBase rg0)
    VMADOT_SS(28, 0, 1)         // vmadot v28, v0, v1          (rg0)
    WORD $0x02028107        // vle8.v v2, (X5=wBase rg1)
    VMADOT_SS(26, 2, 1)         // vmadot v26, v2, v1          (rg1)
    WORD $0x02030207        // vle8.v v4, (X6=wBase rg2)
    VMADOT_SS(24, 4, 1)         // vmadot v24, v4, v1          (rg2)
    WORD $0x02038307        // vle8.v v6, (X7=wBase rg3)
    VMADOT_SS(22, 6, 1)         // vmadot v22, v6, v1          (rg3)
    ADD  $128, X10
    ADD  $128, X5
    ADD  $128, X6
    ADD  $128, X7
    ADD  $128, X12

    // Tile 2 (columns 16-31):
    WORD $0x02060087        // vle8.v v1, (X12)
    WORD $0x02050007        // vle8.v v0, (X10)
    VMADOT_SS(28, 0, 1)         // vmadot v28, v0, v1
    WORD $0x02028107        // vle8.v v2, (X5)
    VMADOT_SS(26, 2, 1)         // vmadot v26, v2, v1
    WORD $0x02030207        // vle8.v v4, (X6)
    VMADOT_SS(24, 4, 1)         // vmadot v24, v4, v1
    WORD $0x02038307        // vle8.v v6, (X7)
    VMADOT_SS(22, 6, 1)         // vmadot v22, v6, v1
    ADD  $128, X10
    ADD  $128, X5
    ADD  $128, X6
    ADD  $128, X7
    ADD  $128, X12

    // === 3. Extract 8 diagonal values from each accumulator → intBuf ===
    WORD $0x01107FD7        // vsetvli t0, zero, e32, m2

    // rg0: vse32.v v28 → scratch, then 8 × (LW X30 + SW X30)
    WORD $0x0206EE27        // vse32.v v28, (X13=scratch)
    WORD $0x0006AF03        // LW X30,   0(X13)
    WORD $0x01E72023        // SW X30,   0(X14=intBuf rg0)
    WORD $0x0206AF03        // LW X30,  32(X13)
    WORD $0x01E72223        // SW X30,   4(X14)
    WORD $0x0406AF03        // LW X30,  64(X13)
    WORD $0x01E72423        // SW X30,   8(X14)
    WORD $0x0606AF03        // LW X30,  96(X13)
    WORD $0x01E72623        // SW X30,  12(X14)
    WORD $0x0806AF03        // LW X30, 128(X13)
    WORD $0x01E72823        // SW X30,  16(X14)
    WORD $0x0A06AF03        // LW X30, 160(X13)
    WORD $0x01E72A23        // SW X30,  20(X14)
    WORD $0x0C06AF03        // LW X30, 192(X13)
    WORD $0x01E72C23        // SW X30,  24(X14)
    WORD $0x0E06AF03        // LW X30, 224(X13)
    WORD $0x01E72E23        // SW X30,  28(X14)
    ADD  $32, X14

    // rg1: vse32.v v26 → scratch
    WORD $0x0206ED27        // vse32.v v26, (X13)
    WORD $0x0006AF03        // LW X30,   0(X13)
    WORD $0x01E8A023        // SW X30,   0(X17=intBuf rg1)
    WORD $0x0206AF03
    WORD $0x01E8A223
    WORD $0x0406AF03
    WORD $0x01E8A423
    WORD $0x0606AF03
    WORD $0x01E8A623
    WORD $0x0806AF03
    WORD $0x01E8A823
    WORD $0x0A06AF03
    WORD $0x01E8AA23
    WORD $0x0C06AF03
    WORD $0x01E8AC23
    WORD $0x0E06AF03
    WORD $0x01E8AE23
    ADD  $32, X17

    // rg2: vse32.v v24 → scratch
    WORD $0x0206EC27        // vse32.v v24, (X13)
    WORD $0x0006AF03
    WORD $0x01EE2023        // SW X30,   0(X28=intBuf rg2)
    WORD $0x0206AF03
    WORD $0x01EE2223
    WORD $0x0406AF03
    WORD $0x01EE2423
    WORD $0x0606AF03
    WORD $0x01EE2623
    WORD $0x0806AF03
    WORD $0x01EE2823
    WORD $0x0A06AF03
    WORD $0x01EE2A23
    WORD $0x0C06AF03
    WORD $0x01EE2C23
    WORD $0x0E06AF03
    WORD $0x01EE2E23
    ADD  $32, X28

    // rg3: vse32.v v22 → scratch
    WORD $0x0206EB27        // vse32.v v22, (X13)
    WORD $0x0006AF03
    WORD $0x01EEA023        // SW X30,   0(X29=intBuf rg3)
    WORD $0x0206AF03
    WORD $0x01EEA223
    WORD $0x0406AF03
    WORD $0x01EEA423
    WORD $0x0606AF03
    WORD $0x01EEA623
    WORD $0x0806AF03
    WORD $0x01EEA823
    WORD $0x0A06AF03
    WORD $0x01EEAA23
    WORD $0x0C06AF03
    WORD $0x01EEAC23
    WORD $0x0E06AF03
    WORD $0x01EEAE23
    ADD  $32, X29

    ADD  $-1, X16
    BNE  X16, X0, subloop_x4

done_x4:
    RET
