#include "textflag.h"

// func vmadotI8GroupsAdd1024(wPacked, actPacked *byte, scratch *int32, out, scale *float32, nGroups, K int)
TEXT ·VmadotI8GroupsAdd1024(SB), NOSPLIT, $0-56
    MOV wPacked+0(FP),  X10
    MOV actPacked+8(FP), X17
    MOV scratch+16(FP), X12
    MOV out+24(FP),     X13
    MOV scale+32(FP),   X16
    MOV nGroups+40(FP), X15
    MOV K+48(FP),       X18
    BEQ X15, X0, done_i8groups_add

group_loop_i8_add:
    MOV X17, X11
    WORD $0x011072d7        // vsetvli t0,zero,e32,m2
    WORD $0x5e003e57        // vmv.v.i v28,0
    WORD $0x000072d7        // vsetvli t0,zero,e8,m1
    SRLI $4, X18, X14       // K/16 native 1024-bit tiles
k_loop_i8_add:
    BEQ X14, X0, k_done_i8_add
    WORD $0x02050007        // vle8.v v0,(X10) tile 0
    WORD $0x02058087        // vle8.v v1,(X11) tile 0
    WORD $0xe2103e2b        // vmadot v28,v1,v0
    ADD $128, X10
    ADD $128, X11
    WORD $0x02050007        // vle8.v v0,(X10) tile 1
    WORD $0x02058087        // vle8.v v1,(X11) tile 1
    WORD $0xe2103e2b        // vmadot v28,v1,v0
    ADD $128, X10
    ADD $128, X11
    WORD $0x02050007        // vle8.v v0,(X10) tile 2
    WORD $0x02058087        // vle8.v v1,(X11) tile 2
    WORD $0xe2103e2b        // vmadot v28,v1,v0
    ADD $128, X10
    ADD $128, X11
    WORD $0x02050007        // vle8.v v0,(X10) tile 3
    WORD $0x02058087        // vle8.v v1,(X11) tile 3
    WORD $0xe2103e2b        // vmadot v28,v1,v0
    ADD $128, X10
    ADD $128, X11
    ADD $-4, X14
    JMP k_loop_i8_add
k_done_i8_add:
    WORD $0x011072d7        // vsetvli t0,zero,e32,m2
    WORD $0x02066e27        // vse32.v v28,(X12)
    WORD $0x00062383        // lw t2,0(X12)
    WORD $0xd003f053        // fcvt.s.w ft0,t2
    WORD $0x00082087        // flw ft1,0(X16)
    WORD $0x10107053        // fmul.s ft0,ft0,ft1
    WORD $0x0006a107        // flw ft2,0(X13)
    WORD $0x00207053        // fadd.s ft0,ft0,ft2
    WORD $0x0006a027        // fsw ft0,0(X13)
    WORD $0x02062383        // lw t2,32(X12)
    WORD $0xd003f053        // fcvt.s.w ft0,t2
    WORD $0x10107053        // fmul.s ft0,ft0,ft1
    WORD $0x0046a107        // flw ft2,4(X13)
    WORD $0x00207053        // fadd.s ft0,ft0,ft2
    WORD $0x0006a227        // fsw ft0,4(X13)
    WORD $0x04062383        // lw t2,64(X12)
    WORD $0xd003f053        // fcvt.s.w ft0,t2
    WORD $0x10107053        // fmul.s ft0,ft0,ft1
    WORD $0x0086a107        // flw ft2,8(X13)
    WORD $0x00207053        // fadd.s ft0,ft0,ft2
    WORD $0x0006a427        // fsw ft0,8(X13)
    WORD $0x06062383        // lw t2,96(X12)
    WORD $0xd003f053        // fcvt.s.w ft0,t2
    WORD $0x10107053        // fmul.s ft0,ft0,ft1
    WORD $0x00c6a107        // flw ft2,12(X13)
    WORD $0x00207053        // fadd.s ft0,ft0,ft2
    WORD $0x0006a627        // fsw ft0,12(X13)
    WORD $0x08062383        // lw t2,128(X12)
    WORD $0xd003f053        // fcvt.s.w ft0,t2
    WORD $0x10107053        // fmul.s ft0,ft0,ft1
    WORD $0x0106a107        // flw ft2,16(X13)
    WORD $0x00207053        // fadd.s ft0,ft0,ft2
    WORD $0x0006a827        // fsw ft0,16(X13)
    WORD $0x0a062383        // lw t2,160(X12)
    WORD $0xd003f053        // fcvt.s.w ft0,t2
    WORD $0x10107053        // fmul.s ft0,ft0,ft1
    WORD $0x0146a107        // flw ft2,20(X13)
    WORD $0x00207053        // fadd.s ft0,ft0,ft2
    WORD $0x0006aa27        // fsw ft0,20(X13)
    WORD $0x0c062383        // lw t2,192(X12)
    WORD $0xd003f053        // fcvt.s.w ft0,t2
    WORD $0x10107053        // fmul.s ft0,ft0,ft1
    WORD $0x0186a107        // flw ft2,24(X13)
    WORD $0x00207053        // fadd.s ft0,ft0,ft2
    WORD $0x0006ac27        // fsw ft0,24(X13)
    WORD $0x0e062383        // lw t2,224(X12)
    WORD $0xd003f053        // fcvt.s.w ft0,t2
    WORD $0x10107053        // fmul.s ft0,ft0,ft1
    WORD $0x01c6a107        // flw ft2,28(X13)
    WORD $0x00207053        // fadd.s ft0,ft0,ft2
    WORD $0x0006ae27        // fsw ft0,28(X13)
    ADD $32, X13            // next 8 float32 outputs
    ADD $-1, X15
    BNE X15, X0, group_loop_i8_add

done_i8groups_add:
    RET
