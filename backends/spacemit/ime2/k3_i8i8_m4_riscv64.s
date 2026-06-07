#include "textflag.h"

// func k3I8I8M4(a *byte, b *byte, c *float32, kBlks int, ldcBytes int)
// Direct port of llama.cpp gemm_kernel_i8i8_m4 for one N32 tile.
TEXT ·K3I8I8M4(SB), NOSPLIT, $0-40
    MOV a+0(FP), X10
    MOV b+8(FP), X11
    MOV c+16(FP), X12
    MOV kBlks+24(FP), X14
    MOV ldcBytes+32(FP), X13

    WORD $0x010072d7        // vsetvli t0, zero, e32, m1
    WORD $0x2fce0e57        // vxor.vv v28, v28, v28
    WORD $0x2fde8ed7        // vxor.vv v29, v29, v29
    WORD $0x2fef0f57        // vxor.vv v30, v30, v30
    WORD $0x2fff8fd7        // vxor.vv v31, v31, v31

loop_i8i8_m4:
    WORD $0x00052507        // flw fa0, 0(X10)
    WORD $0x00452587        // flw fa1, 4(X10)
    WORD $0x00852607        // flw fa2, 8(X10)
    WORD $0x00c52687        // flw fa3, 12(X10)
    ADD  $24, X10

    WORD $0x00f072d7        // vsetvli t0, zero, e16, mf2
    WORD $0x0205d607        // vle16.v v12, (X11)
    ADD  $64, X11
    WORD $0x4ac61757        // vfwcvt.f.f.v v14, v12

    WORD $0x000072d7        // vsetvli t0, zero, e8, m1
    WORD $0x02850007        // vl1r.v v0, (X10)
    ADD  $128, X10
    WORD $0x62858207        // vl4r.v v4, (X11)
    ADD  $512, X11
    WORD $0x62858407        // vl4r.v v8, (X11)
    ADD  $512, X11

    WORD $0x010072d7        // vsetvli t0, zero, e32, m1
    WORD $0x6600512b        // vupack.vv v2, v0, v0, 1
    WORD $0x66525c2b        // vupack.vv v24, v4, v5, 1
    WORD $0x66735d2b        // vupack.vv v26, v6, v7, 1
    WORD $0x6694522b        // vupack.vv v4, v8, v9, 1
    WORD $0x66b5532b        // vupack.vv v6, v10, v11, 1

    WORD $0x2f080857        // vxor.vv v16, v16, v16
    WORD $0x2f080957        // vxor.vv v18, v16, v16
    WORD $0x2f080a57        // vxor.vv v20, v16, v16
    WORD $0x2f080b57        // vxor.vv v22, v16, v16

    WORD $0xe381382b        // vmadot v16, v2, v24
    WORD $0xe3a1392b        // vmadot v18, v2, v26
    WORD $0xe2413a2b        // vmadot v20, v2, v4
    WORD $0xe2613b2b        // vmadot v22, v2, v6
    WORD $0xe391b82b        // vmadot v16, v3, v25
    WORD $0xe3b1b92b        // vmadot v18, v3, v27
    WORD $0xe251ba2b        // vmadot v20, v3, v5
    WORD $0xe271bb2b        // vmadot v22, v3, v7

    WORD $0x6728202b        // vpack.vv v0, v16, v18, 2
    WORD $0x676a212b        // vpack.vv v2, v20, v22, 2
    WORD $0x6620382b        // vpack.vv v16, v0, v2, 3
    WORD $0x6630b92b        // vpack.vv v18, v1, v3, 3

    WORD $0x4b019857        // vfcvt.f.x.v v16, v16
    WORD $0x4b1198d7        // vfcvt.f.x.v v17, v17
    WORD $0x4b219957        // vfcvt.f.x.v v18, v18
    WORD $0x4b3199d7        // vfcvt.f.x.v v19, v19

    WORD $0x93071857        // vfmul.vv v16, v16, v14
    WORD $0x931718d7        // vfmul.vv v17, v17, v14
    WORD $0x93271957        // vfmul.vv v18, v18, v14
    WORD $0x933719d7        // vfmul.vv v19, v19, v14

    WORD $0xb3055e57        // vfmacc.vf v28, fa0, v16
    WORD $0xb315ded7        // vfmacc.vf v29, fa1, v17
    WORD $0xb3265f57        // vfmacc.vf v30, fa2, v18
    WORD $0xb336dfd7        // vfmacc.vf v31, fa3, v19

    ADD  $-1, X14
    BGT  X14, X0, loop_i8i8_m4

    WORD $0x010072d7        // vsetvli t0, zero, e32, m1
    ADD  X13, X12, X7
    WORD $0x02066e27        // vse32.v v28, (X12)
    ADD  X13, X7, X28
    WORD $0x0203eea7        // vse32.v v29, (X7)
    ADD  X13, X28, X7
    WORD $0x020e6f27        // vse32.v v30, (X28)
    WORD $0x0203efa7        // vse32.v v31, (X7)
    RET
