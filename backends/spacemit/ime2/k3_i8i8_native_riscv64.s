#include "textflag.h"

// func k3I8I8M1(a *byte, b *byte, c *float32, kBlks int, nBlks int)
// Direct port of llama.cpp gemm_kernel_i8i8_m1 for one N32 tile.
// A per K32: fp32 scale + int16 sum + 32 int8 q = 38B.
// B per K32/N32: fp16 d[32] + int8 q[32][32] = 1088B.
TEXT ·K3I8I8M1(SB), NOSPLIT, $0-40
    MOV a+0(FP), X18         // A scale base
    MOV a+0(FP), X19         // A data base
    ADD $6, X19
    MOV b+8(FP), X20         // B scale base
    MOV b+8(FP), X21         // B data base
    ADD $64, X21
    MOV c+16(FP), X22
    MOV kBlks+24(FP), X15
    MOV nBlks+32(FP), X28

    WORD $0x010072d7        // vsetvli t0, zero, e32, m1
    WORD $0x2e000157        // vxor.vv v2, v0, v0

k_loop_i8i8_m1:
    WORD $0x000072d7        // vsetvli t0, zero, e8, m1
    WORD $0x628a8207        // vl4r.v v4, (X21)
    ADD  $512, X21
    WORD $0x628a8407        // vl4r.v v8, (X21)
    ADD  $576, X21

    WORD $0x007072d7        // vsetvli t0, zero, e8, mf2
    WORD $0x020a0007        // vle8.v v0, (X20)
    ADD  $1088, X20

    WORD $0x006072d7        // vsetvli t0, zero, e8, mf4
    WORD $0x02098187        // vle8.v v3, (X19)
    ADD  $38, X19
    WORD $0x00092007        // flw f0, 0(X18)
    ADD  $38, X18

    WORD $0x010072d7        // vsetvli t0, zero, e32, m1
    WORD $0x66525c2b        // vupack.vv v24, v4, v5, 1
    WORD $0x66735d2b        // vupack.vv v26, v6, v7, 1
    WORD $0x66945e2b        // vupack.vv v28, v8, v9, 1
    WORD $0x66b55f2b        // vupack.vv v30, v10, v11, 1
    WORD $0x3e323257        // vslidedown.vi v4, v3, 4

    WORD $0x2f080857        // vxor.vv v16, v16, v16
    WORD $0x2f080957        // vxor.vv v18, v16, v16
    WORD $0x2f080a57        // vxor.vv v20, v16, v16
    WORD $0x2f080b57        // vxor.vv v22, v16, v16

    WORD $0xe381b82b        // vmadot v16, v3, v24
    WORD $0xe3a1b92b        // vmadot v18, v3, v26
    WORD $0xe3c1ba2b        // vmadot v20, v3, v28
    WORD $0xe3e1bb2b        // vmadot v22, v3, v30
    WORD $0xe392382b        // vmadot v16, v4, v25
    WORD $0xe3b2392b        // vmadot v18, v4, v27
    WORD $0xe3d23a2b        // vmadot v20, v4, v29
    WORD $0xe3f23b2b        // vmadot v22, v4, v31

    WORD $0x67282c2b        // vpack.vv v24, v16, v18, 2
    WORD $0x676a2d2b        // vpack.vv v26, v20, v22, 2
    WORD $0x67ac382b        // vpack.vv v16, v24, v26, 3

    WORD $0x00f072d7        // vsetvli t0, zero, e16, mf2
    WORD $0x4a061c57        // vfwcvt.f.f.v v24, v0
    WORD $0x010072d7        // vsetvli t0, zero, e32, m1
    WORD $0x4b019d57        // vfcvt.f.x.v v26, v16
    WORD $0x938050d7        // vfmul.vf v1, v24, f0
    WORD $0xb3a09157        // vfmacc.vv v2, v1, v26

    ADD $-1, X15
    BGT X15, X0, k_loop_i8i8_m1

    WORD $0x010e72d7        // vsetvli t0, X28, e32, m1
    WORD $0x020b6127        // vse32.v v2, (X22)
    RET
