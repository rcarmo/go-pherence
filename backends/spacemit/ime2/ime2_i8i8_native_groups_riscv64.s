#include "textflag.h"
#include "ime2_isa.h"

// func k3I8I8M1Groups(a *byte, b *byte, c *float32, kBlks int, nGroups int)
TEXT ·K3I8I8M1Groups(SB), NOSPLIT, $0-40
    MOV b+8(FP), X20          // current group B base
    MOV c+16(FP), X22         // current group C base
    MOV nGroups+32(FP), X28
    BEQ X28, X0, done_i8i8_groups

group_loop_i8i8_groups:
    MOV a+0(FP), X18
    MOV a+0(FP), X19
    ADD $6, X19
    MOV X20, X23              // B scale base for this group
    MOV X20, X21              // B data base for this group
    ADD $64, X21
    MOV kBlks+24(FP), X15

    WORD $0x010072d7        // vsetvli t0, zero, e32, m1
    WORD $0x2e000157        // vxor.vv v2, v0, v0

k_loop_i8i8_groups:
    WORD $0x000072d7        // vsetvli t0, zero, e8, m1
    WORD $0x628a8207        // vl4r.v v4, (X21)
    ADD  $512, X21
    WORD $0x628a8407        // vl4r.v v8, (X21)
    ADD  $576, X21

    WORD $0x007072d7        // vsetvli t0, zero, e8, mf2
    WORD $0x020b8007        // vle8.v v0, (X23)
    ADD  $1088, X23

    WORD $0x006072d7        // vsetvli t0, zero, e8, mf4
    WORD $0x02098187        // vle8.v v3, (X19)
    ADD  $38, X19
    WORD $0x00092007        // flw f0, 0(X18)
    ADD  $38, X18

    WORD $0x010072d7        // vsetvli t0, zero, e32, m1
    VUPACK(24, 4, 5)            // vupack.vv v24, v4, v5, 1
    VUPACK(26, 6, 7)            // vupack.vv v26, v6, v7, 1
    VUPACK(28, 8, 9)            // vupack.vv v28, v8, v9, 1
    VUPACK(30, 10, 11)          // vupack.vv v30, v10, v11, 1
    WORD $0x3e323257        // vslidedown.vi v4, v3, 4

    WORD $0x2f080857        // vxor.vv v16, v16, v16
    WORD $0x2f080957        // vxor.vv v18, v16, v16
    WORD $0x2f080a57        // vxor.vv v20, v16, v16
    WORD $0x2f080b57        // vxor.vv v22, v16, v16

    VMADOT_SS(16, 3, 24)        // vmadot v16, v3, v24
    VMADOT_SS(18, 3, 26)        // vmadot v18, v3, v26
    VMADOT_SS(20, 3, 28)        // vmadot v20, v3, v28
    VMADOT_SS(22, 3, 30)        // vmadot v22, v3, v30
    VMADOT_SS(16, 4, 25)        // vmadot v16, v4, v25
    VMADOT_SS(18, 4, 27)        // vmadot v18, v4, v27
    VMADOT_SS(20, 4, 29)        // vmadot v20, v4, v29
    VMADOT_SS(22, 4, 31)        // vmadot v22, v4, v31

    VPACK(24, 16, 18, 2)        // vpack.vv v24, v16, v18, 2
    VPACK(26, 20, 22, 2)        // vpack.vv v26, v20, v22, 2
    VPACK(16, 24, 26, 3)        // vpack.vv v16, v24, v26, 3

    WORD $0x00f072d7        // vsetvli t0, zero, e16, mf2
    WORD $0x4a061c57        // vfwcvt.f.f.v v24, v0
    WORD $0x010072d7        // vsetvli t0, zero, e32, m1
    WORD $0x4b019d57        // vfcvt.f.x.v v26, v16
    WORD $0x938050d7        // vfmul.vf v1, v24, f0
    WORD $0xb3a09157        // vfmacc.vv v2, v1, v26

    ADD $-1, X15
    BGT X15, X0, k_loop_i8i8_groups

    WORD $0x010072d7        // vsetvli t0, zero, e32, m1
    WORD $0x020b6127        // vse32.v v2, (X22)
    ADD $128, X22
    MOV kBlks+24(FP), X15
    MOV $1088, X24
    MUL X15, X24, X24
    ADD X24, X20
    ADD $-1, X28
    BNE X28, X0, group_loop_i8i8_groups

done_i8i8_groups:
    RET
