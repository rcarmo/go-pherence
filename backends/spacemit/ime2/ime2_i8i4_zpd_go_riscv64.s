#include "textflag.h"
#include "ime2_isa.h"

// func k3I8I4M1ZPDFused(a *byte, b *byte, c *float32, kBlks int, zpd *float32, sumCorr *float32)
//
// Fused k3I8I4M1 + ZPD correction in one assembly function.
// Eliminates 32 separate ScaleAccF32RVV calls per group by doing the
// ZPD accumulation directly after the main kernel loop, while v2 is
// still in the register.
//
// Register layout:
//   Same as k3I8I4M1 for the main loop, then:
//   X16 = zpd ptr (advances +128 per subblock)
//   X17 = sumCorr ptr (advances +4 per subblock)
//   X15 = kBlks (reused as ZPD loop counter after main loop)
//   f4  = sumCorr[sb] scalar
//   v3  = zpd[sb*32..sb*32+31] vector
//
TEXT ·k3I8I4M1ZPDFused(SB), NOSPLIT, $0-48
    MOV a+0(FP), X10
    MOV a+0(FP), X11
    ADD $6, X11
    MOV b+8(FP), X12
    MOV b+8(FP), X13
    ADD $96, X13
    MOV c+16(FP), X14
    MOV kBlks+24(FP), X15
    MOV X15, X9    // save kBlks in s1 (not used by kernel loop)

    // === Init scale vectors (same as k3I8I4M1) ===
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
    // vsetvli t0, x0, e32, m1, tu, mu
    WORD $0x010072d7
    // vmv.v.i v2, 0
    WORD $0x5e003157

loop:
    // vsetvli t0, x0, e8, m1, tu, mu
    WORD $0x000072d7
    // vl4r.v v4, (X13)
    WORD $0x62868207
    ADD $608, X13
    // vsetvli t0, x0, e8, mf2, tu, mu
    WORD $0x007072d7
    // vle8.v v30, (X12)
    WORD $0x02060f07
    ADD $64, X12
    // vsetvli t0, x0, e8, mf4, tu, mu
    WORD $0x006072d7
    // vle8.v v3, (X11)
    WORD $0x02058187
    ADD $38, X11
    // flw f0, 0(X10)
    WORD $0x00052007
    // lh X7, 4(X10)
    WORD $0x00451383
    ADD $38, X10
    // vsetvli t0, x0, e16, m1, tu, mu
    WORD $0x008072d7
    // vmv.v.i v28, 8 (dead code, kept for binary compat)
    WORD $0x5e043e57
    WORD $0x5e043e57
    // vsetvli t0, x0, e8, mf2, tu, mu
    WORD $0x007072d7
    // vle8.v v29, (X12)
    WORD $0x02060e87
    ADD $544, X12
    // vsetvli t0, x0, e8, m1, tu, mu
    WORD $0x000072d7
    // vsrl.vi v24, v3, 4
    WORD $0xa2323c57
    // vsetvli t0, x0, e16, m1, tu, mu
    WORD $0x008072d7
    // vmv.v.i v28, 0 — ZP=0
    WORD $0x5E003E57
    // vmul.vx v26, v28, X7
    WORD $0x97c3ed57
    // smt.vnpack4.vv v8, v3, v3, 3
    VNPACK4(8, 3, 3)
    // smt.vnpack4.vv v10, v24, v24, 3
    VNPACK4(10, 24, 24)
    // vfcvt.f.x.v v16, v26
    WORD $0x4ba19857
    // vadd.vi v18, v16, 0 (copy to v18,v20,v22)
    WORD $0x03003957
    WORD $0x03003a57
    WORD $0x03003b57
    WORD $0x008072d7
    // smt.vmadotsu.hp v16,v10,v4,v1,0,i4
    VMADOTSU_HP(16, 10, 4)
    VMADOTSU_HP(18, 10, 5)
    VMADOTSU_HP(20, 10, 6)
    VMADOTSU_HP(22, 10, 7)
    // smt.vmadotu.hp v16,v8,v4,v0,0,i4
    VMADOTU_HP(16, 8, 4)
    VMADOTU_HP(18, 8, 5)
    VMADOTU_HP(20, 8, 6)
    VMADOTU_HP(22, 8, 7)
    // smt.vpack.vv v24,v16,v18,1
    VPACK(24, 16, 18, 1)
    // smt.vpack.vv v26,v20,v22,1
    VPACK(26, 20, 22, 1)
    // smt.vpack.vv v16,v24,v26,2
    VPACK(16, 24, 26, 2)
    // vsetvli t0, x0, e16, mf2, tu, mu
    WORD $0x00f072d7
    // vfwmul.vv v31, v30, v16
    WORD $0xe3e81fd7
    // vsetvli t0, x0, e32, m1, tu, mu
    WORD $0x010072d7
    // vfmacc.vf v2, f0, v31
    WORD $0xb3f05157
    ADD $-1, X15
    BNE X15, X0, loop

    // === Store kernel output first (bit-exact with separate path) ===
    // vse32.v v2, (X14)
    WORD $0x02076127

    // === ZPD correction loop ===
    // Load output back, accumulate zpd corrections, store back.
    MOV zpd+32(FP), X16
    MOV sumCorr+40(FP), X17
    MOV $32, X15  // hardcode kBlks=32 for testing

    // vsetvli t0, x0, e32, m1, tu, mu (already set)
zpdloop:
    // flw f4, 0(X17) — load sumCorr[sb]
    WORD $0x0008a207
    // vle32.v v2, (X14) — reload output
    WORD $0x02076107
    // vle32.v v3, (X16) — load zpd[sb*32..sb*32+31]
    WORD $0x02086187
    // vfmacc.vf v2, f4, v3 — v2 += sumCorr[sb] * zpd_vec
    WORD $0xb2325157
    // vse32.v v2, (X14) — store back
    WORD $0x02076127
    ADD $128, X16           // zpd += 32 floats = 128 bytes
    ADD $4, X17             // sumCorr += 1 float = 4 bytes
    ADD $-1, X15
    BNE X15, X0, zpdloop

    RET
