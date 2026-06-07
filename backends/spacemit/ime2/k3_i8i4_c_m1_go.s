#include "textflag.h"

// func k3I8I4M1C(a *byte, b *byte, c *float32, kBlks int, nBlks int)
// Faithful C-M1 correction-order clone attempt based on ime2_gemm_i8i4_m1.s
// active path, kept separate from safe k3I8I4M1.
TEXT ·K3I8I4M1C(SB), NOSPLIT, $0-40
    MOV a+0(FP), X10          // A scale/sum ptr
    MOV a+0(FP), X11
    ADD $6, X11               // A q8 data ptr
    MOV b+8(FP), X12          // B scale/zp ptr
    MOV b+8(FP), X13
    ADD $96, X13              // B q4 data ptr
    MOV c+16(FP), X14         // C output ptr
    MOV kBlks+24(FP), X15     // loop counter
    MOV nBlks+32(FP), X6      // final vl

    // C init: v0=1, v1=16, v2=0, convert v0/v1 as original does.
    WORD $0x008072d7        // vsetvli t0, zero, e16,m1,tu,mu
    WORD $0x5e00b057        // vmv.v.i v0, 1
    WORD $0x960230d7        // vsll.vi v1, v0, 4
    WORD $0x2e000157        // vxor.vv v2, v0, v0
    WORD $0x4a019057        // vfcvt.f.x.v v0, v0
    WORD $0x4a1190d7        // vfcvt.f.x.v v1, v1

loop_c_m1:
    // Load B q4 columns, B fp16 scales, B zp, A q8, A scale/sum.
    WORD $0x000072d7        // vsetvli t0, zero, e8,m1,tu,mu
    WORD $0x62868207        // vl4r.v v4, (X13)
    ADD $608, X13

    WORD $0x007072d7        // vsetvli t0, zero, e8,mf2,tu,mu
    WORD $0x02060f07        // vle8.v v30, (X12)  B fp16 d[32] as bytes
    ADD $64, X12

    WORD $0x006072d7        // vsetvli t0, zero, e8,mf4,tu,mu
    WORD $0x02060f87        // vle8.v v31, (X12)  B zp[32]
    ADD $544, X12
    WORD $0x02058187        // vle8.v v3, (X11)   A q8[32]
    ADD $38, X11

    WORD $0x00052007        // flw f0, 0(X10)     A scale
    WORD $0x00451383        // lh X7, 4(X10)      A SumNeg
    ADD $38, X10

    // A low/high unpack.
    WORD $0x000072d7        // vsetvli t0, zero, e8,m1,tu,mu
    WORD $0xa2323c57        // vsrl.vi v24, v3, 4
    WORD $0x008072d7        // vsetvli t0, zero, e16,m1,tu,mu
    WORD $0x4231b42b        // vnpack4.vv v8, v3, v3, 3
    WORD $0x438c352b        // vnpack4.vv v10, v24, v24, 3

    // Dot accumulators start at zero in original active path.
    WORD $0x2f080857        // vxor.vv v16, v16, v16
    WORD $0x2f080957        // vxor.vv v18, v16, v16
    WORD $0x2f080a57        // vxor.vv v20, v16, v16
    WORD $0x2f080b57        // vxor.vv v22, v16, v16

    // hp dot sequence.
    WORD $0xd645082b        // vmadotsu.hp v16,v10,v4,v1,0,i4
    WORD $0xd655092b        // vmadotsu.hp v18,v10,v5,v1,0,i4
    WORD $0xd6650a2b        // vmadotsu.hp v20,v10,v6,v1,0,i4
    WORD $0xd6750b2b        // vmadotsu.hp v22,v10,v7,v1,0,i4
    WORD $0xcc44082b        // vmadotu.hp v16,v8,v4,v0,0,i4
    WORD $0xcc54092b        // vmadotu.hp v18,v8,v5,v0,0,i4
    WORD $0xcc640a2b        // vmadotu.hp v20,v8,v6,v0,0,i4
    WORD $0xcc740b2b        // vmadotu.hp v22,v8,v7,v0,0,i4

    // Original C active-path correction after dots/pack.
    WORD $0x006072d7        // vsetvli t0, zero, e8,mf4,tu,mu
    WORD $0xc3f06e57        // vwcvtu.x.x.v v28, v31  zp -> u16

    WORD $0x000072d7        // vsetvli t0, zero, e8,m1,tu,mu
    WORD $0x67281c2b        // vpack.vv v24,v16,v18,1
    WORD $0x676a1d2b        // vpack.vv v26,v20,v22,1
    WORD $0x67ac282b        // vpack.vv v16,v24,v26,2

    WORD $0x00f072d7        // vsetvli t0, zero, e16,mf2,tu,mu
    WORD $0x97c3ed57        // vmul.vx v26, v28, X7   zp * SumNeg
    WORD $0x4be61b57        // vfwcvt.f.f.v v22, v30  widen B scale
    WORD $0x4ba19957        // vfcvt.f.x.v v18, v26   correction to fp
    WORD $0x008072d7        // vsetvli t0, zero, e16,m1,tu,mu
    WORD $0xc3281a57        // vfwadd.vv v20, v18, v16 dot + correction
    WORD $0x010072d7        // vsetvli t0, zero, e32,m1,tu,mu
    WORD $0x936a1fd7        // vfmul.vv v31, v22, v20 scale * corrected dot
    WORD $0xb3f05157        // vfmacc.vf v2, f0, v31  A scale accumulate

    ADD $-1, X15
    BNE X15, X0, loop_c_m1

    WORD $0x010372d7        // vsetvli t0, X6, e32,m1,tu,mu
    WORD $0x02076127        // vse32.v v2, (X14)
    RET

// func k3I8I4M1CResidual(a *byte, b *byte, residual *float32, c *float32, kBlks int, nBlks int)
// C-M1 clone plus fused exact residual vector: v31 += residual[32] * SumNeg before A-scale vfmacc.
TEXT ·K3I8I4M1CResidual(SB), NOSPLIT, $0-48
    MOV a+0(FP), X10          // A scale/sum ptr
    MOV a+0(FP), X11
    ADD $6, X11               // A q8 data ptr
    MOV b+8(FP), X12          // B scale/zp ptr
    MOV b+8(FP), X13
    ADD $96, X13              // B q4 data ptr
    MOV residual+16(FP), X16  // residual[groups][subs][32] float32, advances +128/subblock
    MOV c+24(FP), X14         // C output ptr
    MOV kBlks+32(FP), X15     // loop counter
    MOV nBlks+40(FP), X6      // final vl

    // C init: v0=1, v1=16, v2=0, convert v0/v1 as original does.
    WORD $0x008072d7        // vsetvli t0, zero, e16,m1,tu,mu
    WORD $0x5e00b057        // vmv.v.i v0, 1
    WORD $0x960230d7        // vsll.vi v1, v0, 4
    WORD $0x2e000157        // vxor.vv v2, v0, v0
    WORD $0x4a019057        // vfcvt.f.x.v v0, v0
    WORD $0x4a1190d7        // vfcvt.f.x.v v1, v1

loop_c_m1_residual:
    // Load B q4 columns, B fp16 scales, B zp, A q8, A scale/sum.
    WORD $0x000072d7        // vsetvli t0, zero, e8,m1,tu,mu
    WORD $0x62868207        // vl4r.v v4, (X13)
    ADD $608, X13

    WORD $0x007072d7        // vsetvli t0, zero, e8,mf2,tu,mu
    WORD $0x02060f07        // vle8.v v30, (X12)  B fp16 d[32] as bytes
    ADD $64, X12

    WORD $0x006072d7        // vsetvli t0, zero, e8,mf4,tu,mu
    WORD $0x02060f87        // vle8.v v31, (X12)  B zp[32]
    ADD $544, X12
    WORD $0x02058187        // vle8.v v3, (X11)   A q8[32]
    ADD $38, X11

    WORD $0x00052007        // flw f0, 0(X10)     A scale
    WORD $0x00451383        // lh X7, 4(X10)      A SumNeg
    ADD $38, X10

    // A low/high unpack.
    WORD $0x000072d7        // vsetvli t0, zero, e8,m1,tu,mu
    WORD $0xa2323c57        // vsrl.vi v24, v3, 4
    WORD $0x008072d7        // vsetvli t0, zero, e16,m1,tu,mu
    WORD $0x4231b42b        // vnpack4.vv v8, v3, v3, 3
    WORD $0x438c352b        // vnpack4.vv v10, v24, v24, 3

    // Dot accumulators start at zero in original active path.
    WORD $0x2f080857        // vxor.vv v16, v16, v16
    WORD $0x2f080957        // vxor.vv v18, v16, v16
    WORD $0x2f080a57        // vxor.vv v20, v16, v16
    WORD $0x2f080b57        // vxor.vv v22, v16, v16

    // hp dot sequence.
    WORD $0xd645082b        // vmadotsu.hp v16,v10,v4,v1,0,i4
    WORD $0xd655092b        // vmadotsu.hp v18,v10,v5,v1,0,i4
    WORD $0xd6650a2b        // vmadotsu.hp v20,v10,v6,v1,0,i4
    WORD $0xd6750b2b        // vmadotsu.hp v22,v10,v7,v1,0,i4
    WORD $0xcc44082b        // vmadotu.hp v16,v8,v4,v0,0,i4
    WORD $0xcc54092b        // vmadotu.hp v18,v8,v5,v0,0,i4
    WORD $0xcc640a2b        // vmadotu.hp v20,v8,v6,v0,0,i4
    WORD $0xcc740b2b        // vmadotu.hp v22,v8,v7,v0,0,i4

    // Original C active-path correction after dots/pack.
    WORD $0x006072d7        // vsetvli t0, zero, e8,mf4,tu,mu
    WORD $0xc3f06e57        // vwcvtu.x.x.v v28, v31  zp -> u16

    WORD $0x000072d7        // vsetvli t0, zero, e8,m1,tu,mu
    WORD $0x67281c2b        // vpack.vv v24,v16,v18,1
    WORD $0x676a1d2b        // vpack.vv v26,v20,v22,1
    WORD $0x67ac282b        // vpack.vv v16,v24,v26,2

    WORD $0x00f072d7        // vsetvli t0, zero, e16,mf2,tu,mu
    WORD $0x97c3ed57        // vmul.vx v26, v28, X7   zp * SumNeg
    WORD $0x4be61b57        // vfwcvt.f.f.v v22, v30  widen B scale
    WORD $0x4ba19957        // vfcvt.f.x.v v18, v26   correction to fp
    WORD $0x008072d7        // vsetvli t0, zero, e16,m1,tu,mu
    WORD $0xc3281a57        // vfwadd.vv v20, v18, v16 dot + correction
    WORD $0x010072d7        // vsetvli t0, zero, e32,m1,tu,mu
    WORD $0x936a1fd7        // vfmul.vv v31, v22, v20 scale * corrected dot

    // Fused exact residual: residual is in float units, so add SumNeg*residual
    // before the scalar A-scale vfmacc.
    WORD $0x02086a07        // vle32.v v20, (X16)       residual[32]
    ADD $128, X16
    WORD $0x5e03c957        // vmv.v.x v18, X7          SumNeg as int32 lanes
    WORD $0x4b219957        // vfcvt.f.x.v v18, v18     SumNeg -> fp32 lanes
    WORD $0x93491a57        // vfmul.vv v20, v20, v18   residual * SumNeg
    WORD $0x03fa1fd7        // vfadd.vv v31, v31, v20   add residual correction

    WORD $0xb3f05157        // vfmacc.vf v2, f0, v31  A scale accumulate

    ADD $-1, X15
    BNE X15, X0, loop_c_m1_residual

    WORD $0x010372d7        // vsetvli t0, X6, e32,m1,tu,mu
    WORD $0x02076127        // vse32.v v2, (X14)
    RET
