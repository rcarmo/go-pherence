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


// func vmadotQ4KScaledLoop1024(wTiles, actBcast *byte, scales, mins, actScale, actSumScaled, out *float32, scratch *int32, numSubs int)
//
// One 8-row Q4_K row-group. The vmadot accumulation, per-row/per-subblock
// scale application, and min correction all happen in RISC-V/SpacemiT assembly:
//   out[r] += dot[r]*scales[r,sb]*actScale[sb] - mins[r,sb]*actSumScaled[sb]
// for each 32-wide subblock.
TEXT ·vmadotQ4KScaledLoop1024(SB), NOSPLIT, $0-72
    MOV wTiles+0(FP),        X10
    MOV actBcast+8(FP),      X11
    MOV scales+16(FP),       X12
    MOV mins+24(FP),         X13
    MOV actScale+32(FP),     X14
    MOV actSumScaled+40(FP), X15
    MOV out+48(FP),          X16
    MOV scratch+56(FP),      X17
    MOV numSubs+64(FP),      X28

    BEQ  X28, X0, done_scaled
    SLL  $2, X28, X31        // row stride in bytes: numSubs * sizeof(float32)

    // Set e32,m2 once so vmv.v.i below uses the right element width.
    WORD $0x011072D7        // vsetvli t0, zero, e32, m2

subloop_scaled:
    // === 1. Zero v28 (64 int32) via vmv.v.i (no memory access) ===
    WORD $0x5E003E57        // vmv.v.i v28, 0

    // === 2. Two vmadot passes (e8, m1 → vl=128) cover one K32 subblock ===
    WORD $0x000072D7        // vsetvli t0, zero, e8, m1

    WORD $0x02050007        // vle8.v v0, (X10)
    WORD $0x02058087        // vle8.v v1, (X11)
    WORD $0xe2103e2b        // vmadot v28, v1, v0
    ADD  $128, X10
    ADD  $128, X11

    WORD $0x02050007        // vle8.v v0, (X10)
    WORD $0x02058087        // vle8.v v1, (X11)
    WORD $0xe2103e2b        // vmadot v28, v1, v0
    ADD  $128, X10
    ADD  $128, X11

    // === 3. Store v28 (256 bytes) to scratch; scalar path extracts diagonals ===
    WORD $0x011072D7        // vsetvli t0, zero, e32, m2
    WORD $0x0208EE27        // vse32.v v28, (X17)

    // === 4. Apply fp32 scaling/min correction for rows 0..7 ===
    WORD $0x00060293        // mv t0, a2  (scale row pointer)
    WORD $0x00068313        // mv t1, a3  (min row pointer)
    WORD $0x0008a383        // lw	t2,0(a7)
    WORD $0xd003f053        // fcvt.s.w	ft0,t2
    WORD $0x0002a087        // flw	ft1,0(t0)
    WORD $0x00072107        // flw	ft2,0(a4)
    WORD $0x10107053        // fmul.s	ft0,ft0,ft1
    WORD $0x10207053        // fmul.s	ft0,ft0,ft2
    WORD $0x00032187        // flw	ft3,0(t1)
    WORD $0x0007a207        // flw	ft4,0(a5)
    WORD $0x1041f1d3        // fmul.s	ft3,ft3,ft4
    WORD $0x08307053        // fsub.s	ft0,ft0,ft3
    WORD $0x00082287        // flw	ft5,0(a6)
    WORD $0x00507053        // fadd.s	ft0,ft0,ft5
    WORD $0x00082027        // fsw	ft0,0(a6)
    WORD $0x01f282b3        // add	t0,t0,t6
    WORD $0x01f30333        // add	t1,t1,t6
    WORD $0x0208a383        // lw	t2,32(a7)
    WORD $0xd003f053        // fcvt.s.w	ft0,t2
    WORD $0x0002a087        // flw	ft1,0(t0)
    WORD $0x00072107        // flw	ft2,0(a4)
    WORD $0x10107053        // fmul.s	ft0,ft0,ft1
    WORD $0x10207053        // fmul.s	ft0,ft0,ft2
    WORD $0x00032187        // flw	ft3,0(t1)
    WORD $0x0007a207        // flw	ft4,0(a5)
    WORD $0x1041f1d3        // fmul.s	ft3,ft3,ft4
    WORD $0x08307053        // fsub.s	ft0,ft0,ft3
    WORD $0x00482287        // flw	ft5,4(a6)
    WORD $0x00507053        // fadd.s	ft0,ft0,ft5
    WORD $0x00082227        // fsw	ft0,4(a6)
    WORD $0x01f282b3        // add	t0,t0,t6
    WORD $0x01f30333        // add	t1,t1,t6
    WORD $0x0408a383        // lw	t2,64(a7)
    WORD $0xd003f053        // fcvt.s.w	ft0,t2
    WORD $0x0002a087        // flw	ft1,0(t0)
    WORD $0x00072107        // flw	ft2,0(a4)
    WORD $0x10107053        // fmul.s	ft0,ft0,ft1
    WORD $0x10207053        // fmul.s	ft0,ft0,ft2
    WORD $0x00032187        // flw	ft3,0(t1)
    WORD $0x0007a207        // flw	ft4,0(a5)
    WORD $0x1041f1d3        // fmul.s	ft3,ft3,ft4
    WORD $0x08307053        // fsub.s	ft0,ft0,ft3
    WORD $0x00882287        // flw	ft5,8(a6)
    WORD $0x00507053        // fadd.s	ft0,ft0,ft5
    WORD $0x00082427        // fsw	ft0,8(a6)
    WORD $0x01f282b3        // add	t0,t0,t6
    WORD $0x01f30333        // add	t1,t1,t6
    WORD $0x0608a383        // lw	t2,96(a7)
    WORD $0xd003f053        // fcvt.s.w	ft0,t2
    WORD $0x0002a087        // flw	ft1,0(t0)
    WORD $0x00072107        // flw	ft2,0(a4)
    WORD $0x10107053        // fmul.s	ft0,ft0,ft1
    WORD $0x10207053        // fmul.s	ft0,ft0,ft2
    WORD $0x00032187        // flw	ft3,0(t1)
    WORD $0x0007a207        // flw	ft4,0(a5)
    WORD $0x1041f1d3        // fmul.s	ft3,ft3,ft4
    WORD $0x08307053        // fsub.s	ft0,ft0,ft3
    WORD $0x00c82287        // flw	ft5,12(a6)
    WORD $0x00507053        // fadd.s	ft0,ft0,ft5
    WORD $0x00082627        // fsw	ft0,12(a6)
    WORD $0x01f282b3        // add	t0,t0,t6
    WORD $0x01f30333        // add	t1,t1,t6
    WORD $0x0808a383        // lw	t2,128(a7)
    WORD $0xd003f053        // fcvt.s.w	ft0,t2
    WORD $0x0002a087        // flw	ft1,0(t0)
    WORD $0x00072107        // flw	ft2,0(a4)
    WORD $0x10107053        // fmul.s	ft0,ft0,ft1
    WORD $0x10207053        // fmul.s	ft0,ft0,ft2
    WORD $0x00032187        // flw	ft3,0(t1)
    WORD $0x0007a207        // flw	ft4,0(a5)
    WORD $0x1041f1d3        // fmul.s	ft3,ft3,ft4
    WORD $0x08307053        // fsub.s	ft0,ft0,ft3
    WORD $0x01082287        // flw	ft5,16(a6)
    WORD $0x00507053        // fadd.s	ft0,ft0,ft5
    WORD $0x00082827        // fsw	ft0,16(a6)
    WORD $0x01f282b3        // add	t0,t0,t6
    WORD $0x01f30333        // add	t1,t1,t6
    WORD $0x0a08a383        // lw	t2,160(a7)
    WORD $0xd003f053        // fcvt.s.w	ft0,t2
    WORD $0x0002a087        // flw	ft1,0(t0)
    WORD $0x00072107        // flw	ft2,0(a4)
    WORD $0x10107053        // fmul.s	ft0,ft0,ft1
    WORD $0x10207053        // fmul.s	ft0,ft0,ft2
    WORD $0x00032187        // flw	ft3,0(t1)
    WORD $0x0007a207        // flw	ft4,0(a5)
    WORD $0x1041f1d3        // fmul.s	ft3,ft3,ft4
    WORD $0x08307053        // fsub.s	ft0,ft0,ft3
    WORD $0x01482287        // flw	ft5,20(a6)
    WORD $0x00507053        // fadd.s	ft0,ft0,ft5
    WORD $0x00082a27        // fsw	ft0,20(a6)
    WORD $0x01f282b3        // add	t0,t0,t6
    WORD $0x01f30333        // add	t1,t1,t6
    WORD $0x0c08a383        // lw	t2,192(a7)
    WORD $0xd003f053        // fcvt.s.w	ft0,t2
    WORD $0x0002a087        // flw	ft1,0(t0)
    WORD $0x00072107        // flw	ft2,0(a4)
    WORD $0x10107053        // fmul.s	ft0,ft0,ft1
    WORD $0x10207053        // fmul.s	ft0,ft0,ft2
    WORD $0x00032187        // flw	ft3,0(t1)
    WORD $0x0007a207        // flw	ft4,0(a5)
    WORD $0x1041f1d3        // fmul.s	ft3,ft3,ft4
    WORD $0x08307053        // fsub.s	ft0,ft0,ft3
    WORD $0x01882287        // flw	ft5,24(a6)
    WORD $0x00507053        // fadd.s	ft0,ft0,ft5
    WORD $0x00082c27        // fsw	ft0,24(a6)
    WORD $0x01f282b3        // add	t0,t0,t6
    WORD $0x01f30333        // add	t1,t1,t6
    WORD $0x0e08a383        // lw	t2,224(a7)
    WORD $0xd003f053        // fcvt.s.w	ft0,t2
    WORD $0x0002a087        // flw	ft1,0(t0)
    WORD $0x00072107        // flw	ft2,0(a4)
    WORD $0x10107053        // fmul.s	ft0,ft0,ft1
    WORD $0x10207053        // fmul.s	ft0,ft0,ft2
    WORD $0x00032187        // flw	ft3,0(t1)
    WORD $0x0007a207        // flw	ft4,0(a5)
    WORD $0x1041f1d3        // fmul.s	ft3,ft3,ft4
    WORD $0x08307053        // fsub.s	ft0,ft0,ft3
    WORD $0x01c82287        // flw	ft5,28(a6)
    WORD $0x00507053        // fadd.s	ft0,ft0,ft5
    WORD $0x00082e27        // fsw	ft0,28(a6)

    ADD  $4, X12             // next subblock scale column
    ADD  $4, X13             // next subblock min column
    ADD  $4, X14             // next actScale
    ADD  $4, X15             // next actSumScaled
    ADD  $-1, X28
    BNE  X28, X0, subloop_scaled

done_scaled:
    RET
