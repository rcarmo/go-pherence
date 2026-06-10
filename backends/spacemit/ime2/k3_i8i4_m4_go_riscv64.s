#include "textflag.h"

// func k3I8I4M4(a *byte, b *byte, c *float32, kBlks int, ldcBytes int)
//
// Native M4 port of llama.cpp gemm_kernel_i8i4_m4 (.hp path).
// Processes 4 A rows × 32 B columns using smt.vmadotsu/u.hp at VLEN=1024.
//
// A layout per K32 block (152 bytes):
//   [fp32 scale[4]: 16B][int16 sum[4]: 8B][int8 q[4][32]: 128B]
//
// B layout per K32 subblock (608 bytes):
//   [fp16_d × 32: 64B][int8_zp × 32: 32B][nibbles × 512B]
//
// Register allocation:
//   X10 = A ptr (advances +152 per k-block)
//   X11 = B scale ptr (base B, advances +608 per k-block total)
//   X12 = B data ptr (B+96, advances +608 per k-block)
//   X13 = C ptr (row 0, static)
//   X14 = kBlks countdown
//   X15 = ldcBytes (C row stride in bytes)
//   X16 = scratch (A quants ptr)
//
//   f0-f3 (ft0-ft3) = A scales for 4 rows
//   v0  = fp16(1.0)  scale for lo4 vmadotu.hp
//   v1  = fp16(16.0) scale for hi4 vmadotsu.hp
//   v14 = B scale replicated to 64 fp16 elements
//   v28 = fp32 accumulator row 0
//   v29 = fp32 accumulator row 1
//   v30 = fp32 accumulator row 2
//   v31 = fp32 accumulator row 3
//
TEXT ·K3I8I4M4(SB), NOSPLIT, $0-40
    MOV a+0(FP), X10
    MOV b+8(FP), X11
    MOV b+8(FP), X12
    ADD $96, X12
    MOV c+16(FP), X13
    MOV kBlks+24(FP), X14
    MOV ldcBytes+32(FP), X15

    // === Init fp16 scale vectors (e16, m1) ===
    // vsetvli t0, x0, e16, m1, tu, mu
    WORD $0x008072d7
    // lui X5, 4 → 0x4000
    WORD $0x000042b7
    // addi X5, X5, -0x400 → 0x3C00 = fp16(1.0)
    WORD $0xc0028293
    // vmv.v.x v0, X5
    WORD $0x5e02c057
    // lui X5, 5 → 0x5000
    WORD $0x000052b7
    // addi X5, X5, -0x400 → 0x4C00 = fp16(16.0)
    WORD $0xc0028293
    // vmv.v.x v1, X5
    WORD $0x5e02c0d7

    // === Zero fp32 accumulators (e32, m1) ===
    // vsetvli t0, x0, e32, m1, tu, mu
    WORD $0x010072d7
    // vmv.v.i v28, 0
    WORD $0x5e003e57
    // vmv.v.i v29, 0
    WORD $0x5e003ed7
    // vmv.v.i v30, 0
    WORD $0x5e003f57
    // vmv.v.i v31, 0
    WORD $0x5e003fd7

kloop:
    // ===== Load B nibbles (512B) into v4-v7 =====
    // vsetvli t0, x0, e8, m1, tu, mu
    WORD $0x000072d7
    // vl4r.v v4, (X12)
    WORD $0x62860207
    ADD $608, X12

    // ===== Load B fp16 scales (64B → 32 fp16) and replicate to 64 =====
    // vsetvli t0, x0, e16, mf2, tu, mu
    WORD $0x00f072d7
    // vle16.v v12, (X11)
    WORD $0x0205d607
    ADD $64, X11
    // vsetvli t0, x0, e16, m1, tu, mu
    WORD $0x008072d7
    // smt.vpack.vv v14, v12, v12, 3 — replicate 32→64 fp16
    WORD $0x66c6372b
    // Skip B zp (32B); ZP=0 with Go correction
    ADD $544, X11

    // ===== Load 4 A scales (fp32) at A+0 =====
    // flw f0, 0(X10) — scale row 0
    WORD $0x00052007
    // flw f1, 4(X10) — scale row 1
    WORD $0x00452087
    // flw f2, 8(X10) — scale row 2
    WORD $0x00852107
    // flw f3, 12(X10) — scale row 3
    WORD $0x00c52187

    // ===== Load 128B A quants at A+24 =====
    MOV X10, X16
    ADD $24, X16
    // vl1r.v v16, (X16)
    WORD $0x02880807
    // Advance A to next block
    ADD $152, X10

    // ===== Nibble extraction (e8, m1) =====
    // vsetvli t0, x0, e8, m1, tu, mu
    WORD $0x000072d7
    // vsrl.vi v17, v16, 4  — hi nibbles
    WORD $0xa30238d7
    // smt.vnpack4.vv v12, v16, v17, 3
    WORD $0x4318362b
    // smt.vupack.vv v2, v12, v12, 2 → v2=lo, v3=hi
    WORD $0x66c6612b

    // ===== Zero per-k dot accumulators (ZP=0) =====
    // vsetvli t0, x0, e16, m1, tu, mu
    WORD $0x008072d7
    // vmv.v.i v18, 0
    WORD $0x5e003957
    // vmv.v.i v19, 0
    WORD $0x5e0039d7
    // vmv.v.i v20, 0
    WORD $0x5e003a57
    // vmv.v.i v21, 0
    WORD $0x5e003ad7

    // ===== hi4 dot products (signed×unsigned, scale=16) =====
    // smt.vmadotsu.hp v18, v3, v4, v1, 0, i4
    WORD $0xd641892b
    // smt.vmadotsu.hp v19, v3, v5, v1, 0, i4
    WORD $0xd65189ab
    // smt.vmadotsu.hp v20, v3, v6, v1, 0, i4
    WORD $0xd6618a2b
    // smt.vmadotsu.hp v21, v3, v7, v1, 0, i4
    WORD $0xd6718aab

    // ===== lo4 dot products (unsigned×unsigned, scale=1) =====
    // smt.vmadotu.hp v18, v2, v4, v0, 0, i4
    WORD $0xcc41092b
    // smt.vmadotu.hp v19, v2, v5, v0, 0, i4
    WORD $0xcc5109ab
    // smt.vmadotu.hp v20, v2, v6, v0, 0, i4
    WORD $0xcc610a2b
    // smt.vmadotu.hp v21, v2, v7, v0, 0, i4
    WORD $0xcc710aab

    // ===== Pack dot results =====
    // smt.vpack.vv v8, v18, v19, 1
    WORD $0x6739142b
    // smt.vpack.vv v12, v20, v21, 1
    WORD $0x675a162b
    // smt.vpack.vv v20, v8, v12, 2 → v20(rows 0,1 fp16), v21(rows 2,3 fp16)
    WORD $0x66c42a2b

    // ===== Widen fp16→fp32 and multiply by B scale =====
    // (still at e16, m1)
    // vfwmul.vv v16, v20, v14 → v16=row0, v17=row1 (fp32)
    WORD $0xe3471857
    // vfwmul.vv v18, v21, v14 → v18=row2, v19=row3 (fp32)
    WORD $0xe3571957

    // ===== Accumulate with A scales (e32, m1) =====
    // vsetvli t0, x0, e32, m1, tu, mu
    WORD $0x010072d7
    // vfmacc.vf v28, ft0, v16
    WORD $0xb3005e57
    // vfmacc.vf v29, ft1, v17
    WORD $0xb310ded7
    // vfmacc.vf v30, ft2, v18
    WORD $0xb3215f57
    // vfmacc.vf v31, ft3, v19
    WORD $0xb331dfd7

    // ===== Loop =====
    ADD $-1, X14
    BNE X14, X0, kloop

    // ===== Store 4 rows of 32 fp32 =====
    // vsetvli t0, x0, e32, m1, tu, mu (already set from loop body)
    // Row 0: vse32.v v28, (X13)
    WORD $0x0206ee27
    // Row 1: X5 = X13 + ldcBytes
    MOV X13, X5
    ADD X15, X5
    // vse32.v v29, (X5)
    WORD $0x0202eea7
    // Row 2: X6 = X5 + ldcBytes
    MOV X5, X6
    ADD X15, X6
    // vse32.v v30, (X6)
    WORD $0x02036f27
    // Row 3: X7 = X6 + ldcBytes
    MOV X6, X7
    ADD X15, X7
    // vse32.v v31, (X7)
    WORD $0x0203efa7
    RET
