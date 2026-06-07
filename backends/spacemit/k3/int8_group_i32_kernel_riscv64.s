#include "textflag.h"

// func vmadotI8GroupsI321024(wPacked, actPacked *byte, scratch, out *int32, nGroups, K int)
TEXT ·vmadotI8GroupsI321024(SB), NOSPLIT, $0-48
    MOV wPacked+0(FP),   X10
    MOV actPacked+8(FP), X17
    MOV scratch+16(FP),  X12
    MOV out+24(FP),      X13
    MOV nGroups+32(FP),  X19
    MOV K+40(FP),        X18
    BEQ X19, X0, done_i8groups_i32

group_loop_i8_i32:
    MOV X17, X11
    WORD $0x011072d7        // vsetvli t0,zero,e32,m2
    WORD $0x5e003e57        // vmv.v.i v28,0
    WORD $0x000072d7        // vsetvli t0,zero,e8,m1
    SRLI $4, X18, X14       // K/16 native 1024-bit tiles
k_loop_i8_i32:
    BEQ X14, X0, k_done_i8_i32
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
    JMP k_loop_i8_i32
k_done_i8_i32:
    WORD $0x011072d7        // vsetvli t0,zero,e32,m2
    WORD $0x02066e27        // vse32.v v28,(X12)
    WORD $0x00062283        // lw X5,0(X12)
    WORD $0x02062303        // lw X6,32(X12)
    WORD $0x04062383        // lw X7,64(X12)
    WORD $0x06062783        // lw X15,96(X12)
    WORD $0x08062803        // lw X16,128(X12)
    WORD $0x0a062e03        // lw X28,160(X12)
    WORD $0x0c062e83        // lw X29,192(X12)
    WORD $0x0e062f03        // lw X30,224(X12)
    WORD $0x0056a023        // sw row 0 to out
    WORD $0x0066a223        // sw row 1 to out
    WORD $0x0076a423        // sw row 2 to out
    WORD $0x00f6a623        // sw row 3 to out
    WORD $0x0106a823        // sw row 4 to out
    WORD $0x01c6aa23        // sw row 5 to out
    WORD $0x01d6ac23        // sw row 6 to out
    WORD $0x01e6ae23        // sw row 7 to out
    ADD $32, X13            // next 8 int32 outputs
    ADD $-1, X19
    BNE X19, X0, group_loop_i8_i32

done_i8groups_i32:
    RET
