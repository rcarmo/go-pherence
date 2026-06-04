#include "textflag.h"

// func k3I8I4M1(a *byte, b *byte, c *float32, kBlks int, nBlks int)
//
// Register layout during loop:
//   X10 = A scale+sum ptr  [fp32_scale:4][int16_sum:2] per subblk, advances +38
//   X11 = A data ptr       [int8_data:32] = A+6, advances +38
//   X12 = B scale ptr      [fp16_d:64][int8_zp:32] per subblk, advances +64 then +544
//   X13 = B data ptr       [nibbles:512] = B+96, advances +608
//   X14 = C output ptr     32×fp32, static
//   X15 = kBlks countdown  (loop counter)
//   X6  = nBlks            (used by epilogue vsetvli to set vl=nBlks for store)
//   v0  = {1,1,...} int16  (scale for lo4 nibbles, vmadotu.hp)
//   v1  = {16,16,...} int16 (scale for hi4 nibbles, vmadotsu.hp)
//   v2  = fp32 accumulator (32 output values)
//
// B sub-block layout (608 bytes):
//   [fp16_d × 32: 64B][int8_zp × 32: 32B][nibbles × 512B]
//   X12 → d, X13 → nibbles (= B+96)
//
// A sub-block layout (38 bytes):
//   [fp32_scale: 4B][int16_sum: 2B][int8_data: 32B]
//   X10 → scale, X11 → data (= A+6)
//
// Changes from broken original:
//   - vxor.vv v2,v0,v0 → vmv.v.i v2,0  (vxor hangs on A100)
//   - removed vfcvt.f.x.v for v0,v1   (vmadotsu/u.hp uses int16 scale, not fp16)
//   - vfcvt.f.x.v v16,v28 → v16,v26   (v26 has vmul result, v28 is ZP not product)
//   - ADD $128,X13 → ADD $96,X13       (nibbles at B+96, not B+128)
//   - ADD $640,X13 → ADD $608,X13      (608-byte sub-block stride)
//   - ADD $576,X12 → ADD $544,X12      (64+544=608 total B scale stride)
TEXT ·k3I8I4M1(SB), NOSPLIT, $0-40
    MOV a+0(FP), X10
    MOV a+0(FP), X11
    ADD $6, X11
    MOV b+8(FP), X12
    MOV b+8(FP), X13
    ADD $96, X13
    MOV c+16(FP), X14
    MOV kBlks+24(FP), X15
    MOV nBlks+32(FP), X6
    // vsetvli t0, x0, e16, m1, tu, mu  — set for vmv.v.x init
    WORD $0x008072d7
    // lui X5, 4     — X5 = 0x4000
    WORD $0x000042b7
    // addi X5, X5, -0x400  — X5 = 0x3C00 = fp16(1.0) bit pattern
    WORD $0xc0028293
    // vmv.v.x v0, X5  — v0 = fp16(1.0) in every e16 element (scale for lo4 vmadotu.hp)
    WORD $0x5e02c057
    // lui X5, 5     — X5 = 0x5000
    WORD $0x000052b7
    // addi X5, X5, -0x400  — X5 = 0x4C00 = fp16(16.0) bit pattern
    WORD $0xc0028293
    // vmv.v.x v1, X5  — v1 = fp16(16.0) in every e16 element (scale for hi4 vmadotsu.hp)
    WORD $0x5e02c0d7
    // vsetvli t0, x0, e32, m1, tu, mu  — set for vmv.v.i v2 zero
    WORD $0x010072d7
    // vmv.v.i v2, 0   (zero fp32 accumulator; safe on A100, unlike vxor.vv)
    WORD $0x5e003157
loop:
    // vsetvli t0, x0, e8, m1, tu, mu
    WORD $0x000072d7
    // vl4r.v v4, (X13)   — load 512B of 4-bit B nibbles into v4-v7
    WORD $0x62868207
    ADD $608, X13
    // vsetvli t0, x0, e8, mf2, tu, mu
    WORD $0x007072d7
    // vle8.v v30, (X12)  — load 64B of B fp16 scales
    WORD $0x02060f07
    ADD $64, X12
    // vsetvli t0, x0, e8, mf4, tu, mu
    WORD $0x006072d7
    // vle8.v v3, (X11)   — load 32 int8 A activations
    WORD $0x02058187
    ADD $38, X11
    // flw f0, 0(X10)     — load A fp32 scale
    WORD $0x00052007
    // lh X7, 4(X10)      — load A int16 sum (= -sumNeg)
    WORD $0x00451383
    ADD $38, X10
    // vsetvli t0, x0, e16, m1, tu, mu
    WORD $0x008072d7
    // vmv.v.i v28, 8  (dead code — overwritten by vwcvtu below; kept for binary compatibility)
    WORD $0x5e043e57
    // vmv.v.i v28, 8  (dead code, kept for binary compatibility)
    // vmv.v.i v28, 8  (dead code — overwritten by vmv.v.i v28,0 below)
    WORD $0x5e043e57
    // vsetvli t0, x0, e8, mf2, tu, mu
    WORD $0x007072d7
    // vle8.v v29, (X12)  -- load B int8 zp bytes
    WORD $0x02060e87
    ADD $544, X12
    // vsetvli t0, x0, e8, m1, tu, mu
    WORD $0x000072d7
    // vsrl.vi v24, v3, 4  -- extract hi nibbles of A
    WORD $0xa2323c57
    // vsetvli t0, x0, e16, m1, tu, mu
    WORD $0x008072d7
    // vmv.v.i v28, 0  -- ZP=0; Go ZPD correction loop applies exact ZP
    WORD $0x5E003E57
    // vmul.vx v26, v28, X7   — v26 = zp_int16 × A_sum
    WORD $0x97c3ed57
    // smt.vnpack4.vv v8, v3, v3, 3   — unpack A lo nibbles → v8
    WORD $0x4231b42b
    // smt.vnpack4.vv v10, v24, v24, 3 — unpack A hi nibbles → v10
    WORD $0x438c352b
    // vfcvt.f.x.v v16, v26  — convert zp×sum to fp16 (ZP correction init for accumulators)
    WORD $0x4ba19857
    // vadd.vi v18, v16, 0   — copy v16 to v18,v20,v22 (init accumulators for groups 8-15,16-23,24-31)
    WORD $0x03003957
    WORD $0x03003a57
    WORD $0x03003b57
    WORD $0x008072d7
    // smt.vmadotsu.hp v16,v10,v4,v1,0,i4  — group cols 0-7:  hi4×B_lo, scale v1=16
    WORD $0xd645082b
    // smt.vmadotsu.hp v18,v10,v5,v1,0,i4  — group cols 8-15
    WORD $0xd655092b
    // smt.vmadotsu.hp v20,v10,v6,v1,0,i4  — group cols 16-23
    WORD $0xd6650a2b
    // smt.vmadotsu.hp v22,v10,v7,v1,0,i4  — group cols 24-31
    WORD $0xd6750b2b
    // smt.vmadotu.hp v16,v8,v4,v0,0,i4   — lo4×B_lo, scale v0=1
    WORD $0xcc44082b
    WORD $0xcc54092b
    WORD $0xcc640a2b
    WORD $0xcc740b2b
    // smt.vpack.vv v24,v16,v18,1   — pack groups 0-7 and 8-15 → v24
    WORD $0x67281c2b
    // smt.vpack.vv v26,v20,v22,1   — pack groups 16-23 and 24-31 → v26
    WORD $0x676a1d2b
    // smt.vpack.vv v16,v24,v26,2   — pack all 32 cols → v16 (fp16)
    WORD $0x67ac282b
    // vsetvli t0, x0, e16, mf2, tu, mu
    WORD $0x00f072d7
    // vfwmul.vv v31, v30, v16  — fp16×fp16→fp32: B_scale × dot_result
    WORD $0xe3e81fd7
    // vsetvli t0, x0, e32, m1, tu, mu
    WORD $0x010072d7
    // vfmacc.vf v2, f0, v31   — v2 += A_scale × v31 (fp32 accumulate)
    WORD $0xb3f05157
    ADD $-1, X15
    BNE X15, X0, loop
    // vsetvli t0, X6, e32, m1, tu, mu  — set vl = nBlks (=32) for output store
    WORD $0x010372d7
    // vse32.v v2, (X14)   — store 32 fp32 outputs
    WORD $0x02076127
    RET
