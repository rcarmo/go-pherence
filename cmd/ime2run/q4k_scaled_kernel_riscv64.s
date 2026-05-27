#include "textflag.h"

// func vmadotQ4KIntLoop1024(wTiles, actBcast *byte, scratch, intBuf *int32, numSubs int)
//
// scratch: [64]int32 scratch buffer (overwritten each subblock by vse32.v).
//
// For each subblock sb = 0 .. numSubs-1:
//   1. vmv.v.i v28, 0  — zero accumulator (uses e32,m2 vtype set at entry).
//   2. Two vmadot passes over 32 input columns.
//   3. vse32.v v28, (scratch) — stores 64 int32 results.
//   4. 8 scalar LW at stride 32 from scratch → rows 0..7.
//   5. 8 scalar SW to intBuf; advance intBuf by 32 bytes.
//
// Registers:
//   X10 (a0) = wTiles    (advances 256/subblock)
//   X11 (a1) = actBcast  (advances 256/subblock)
//   X12 (a2) = scratch   (static, overwritten each iter)
//   X13 (a3) = intBuf    (advances 32/subblock)
//   X14 (a4) = numSubs   (countdown)
//   X5,X6,X7,X15,X16,X28,X29,X30 = extracted rows (all caller-saved)

TEXT ·vmadotQ4KIntLoop1024(SB), NOSPLIT, $0-40
    MOV wTiles+0(FP),   X10
    MOV actBcast+8(FP), X11
    MOV scratch+16(FP), X12
    MOV intBuf+24(FP),  X13
    MOV numSubs+32(FP), X14

    BEQ  X14, X0, done_int

    // Set e32,m2 once so vmv.v.i below uses the right element width.
    WORD $0x011072D7        // vsetvli t0, zero, e32, m2

subloop_int:
    // === 1. Zero v28 (64 int32) via vmv.v.i (no memory access) ===
    // vmv.v.i vd, simm5=0: funct6=010111, vm=1, vs2=0, simm5=0, vd=v28
    WORD $0x5E003E57        // vmv.v.i v28, 0

    // === 2. Two vmadot passes (e8, m1 → vl=128) ===
    WORD $0x000072D7        // vsetvli t0, zero, e8, m1

    // Tile 1:
    WORD $0x02050007        // vle8.v v0, (a0=X10)
    WORD $0x02058087        // vle8.v v1, (a1=X11)
    WORD $0xe2103e2b        // vmadot v28, v1, v0
    ADD  $128, X10
    ADD  $128, X11

    // Tile 2:
    WORD $0x02050007        // vle8.v v0, (a0)
    WORD $0x02058087        // vle8.v v1, (a1)
    WORD $0xe2103e2b        // vmadot v28, v1, v0
    ADD  $128, X10
    ADD  $128, X11

    // === 3. Store v28 (256 bytes) to scratch; restore e32,m2 ===
    WORD $0x011072D7        // vsetvli t0, zero, e32, m2
    WORD $0x02066E27        // vse32.v v28, (a2=X12)

    // === 4. Scalar LW: 8 diagonal values at stride 32 from scratch (rs1=X12) ===
    WORD $0x00062283        // LW X5,   0(X12)   row 0
    WORD $0x02062303        // LW X6,  32(X12)   row 1
    WORD $0x04062383        // LW X7,  64(X12)   row 2
    WORD $0x06062783        // LW X15, 96(X12)   row 3
    WORD $0x08062803        // LW X16,128(X12)   row 4
    WORD $0x0A062E03        // LW X28,160(X12)   row 5
    WORD $0x0C062E83        // LW X29,192(X12)   row 6
    WORD $0x0E062F03        // LW X30,224(X12)   row 7

    // === 5. SW to intBuf (rs1=X13), advance by 32 bytes ===
    WORD $0x0056A023        // SW X5,  0(X13)   row 0
    WORD $0x0066A223        // SW X6,  4(X13)   row 1
    WORD $0x0076A423        // SW X7,  8(X13)   row 2
    WORD $0x00F6A623        // SW X15,12(X13)   row 3
    WORD $0x0106A823        // SW X16,16(X13)   row 4
    WORD $0x01C6AA23        // SW X28,20(X13)   row 5
    WORD $0x01D6AC23        // SW X29,24(X13)   row 6
    WORD $0x01E6AE23        // SW X30,28(X13)   row 7
    ADD  $32, X13

    ADD  $-1, X14
    BNE  X14, X0, subloop_int

done_int:
    RET
