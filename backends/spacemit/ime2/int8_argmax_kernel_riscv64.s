#include "textflag.h"

// func vmadotI8ArgmaxGroups1024(wPacked, actPacked *byte, scratch *int32, bestVal *int32, bestID *int64, nGroups, K, rowStart int)
TEXT ·VmadotI8ArgmaxGroups1024(SB), NOSPLIT, $0-64
    MOV wPacked+0(FP),   X10
    MOV actPacked+8(FP), X17
    MOV scratch+16(FP),  X12
    MOV bestVal+24(FP),  X13
    MOV bestID+32(FP),   X16
    MOV nGroups+40(FP),  X19
    MOV K+48(FP),        X18
    MOV rowStart+56(FP), X22
    MOV $-2147483648, X21
    MOV X22, X23
    BEQ X19, X0, done_argmax_i8

group_loop_argmax_i8:
    MOV X17, X11
    WORD $0x011072d7        // vsetvli t0,zero,e32,m2
    WORD $0x5e003e57        // vmv.v.i v28,0
    WORD $0x000072d7        // vsetvli t0,zero,e8,m1
    SRLI $4, X18, X14       // K/16 native 1024-bit tiles
k_loop_argmax_i8:
    BEQ X14, X0, k_done_argmax_i8
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
    JMP k_loop_argmax_i8
k_done_argmax_i8:
    WORD $0x011072d7        // vsetvli t0,zero,e32,m2
    WORD $0x02066e27        // vse32.v v28,(X12)
    WORD $0x00062283        // lw X5,0(X12)
    BGT X5, X21, arg_update_0
    JMP arg_skip_0
arg_update_0:
    MOV X5, X21
    ADD $0, X22, X23
arg_skip_0:
    WORD $0x02062283        // lw X5,32(X12)
    BGT X5, X21, arg_update_1
    JMP arg_skip_1
arg_update_1:
    MOV X5, X21
    ADD $1, X22, X23
arg_skip_1:
    WORD $0x04062283        // lw X5,64(X12)
    BGT X5, X21, arg_update_2
    JMP arg_skip_2
arg_update_2:
    MOV X5, X21
    ADD $2, X22, X23
arg_skip_2:
    WORD $0x06062283        // lw X5,96(X12)
    BGT X5, X21, arg_update_3
    JMP arg_skip_3
arg_update_3:
    MOV X5, X21
    ADD $3, X22, X23
arg_skip_3:
    WORD $0x08062283        // lw X5,128(X12)
    BGT X5, X21, arg_update_4
    JMP arg_skip_4
arg_update_4:
    MOV X5, X21
    ADD $4, X22, X23
arg_skip_4:
    WORD $0x0a062283        // lw X5,160(X12)
    BGT X5, X21, arg_update_5
    JMP arg_skip_5
arg_update_5:
    MOV X5, X21
    ADD $5, X22, X23
arg_skip_5:
    WORD $0x0c062283        // lw X5,192(X12)
    BGT X5, X21, arg_update_6
    JMP arg_skip_6
arg_update_6:
    MOV X5, X21
    ADD $6, X22, X23
arg_skip_6:
    WORD $0x0e062283        // lw X5,224(X12)
    BGT X5, X21, arg_update_7
    JMP arg_skip_7
arg_update_7:
    MOV X5, X21
    ADD $7, X22, X23
arg_skip_7:
    ADD $8, X22
    ADD $-1, X19
    BNE X19, X0, group_loop_argmax_i8

done_argmax_i8:
    WORD $0x0156a023        // sw X21,0(X13)
    WORD $0x01783023        // sd X23,0(X16)
    RET
