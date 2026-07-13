#include "textflag.h"

// func ddotAsm(x, y []float64) float64
TEXT ·ddotAsm(SB), NOSPLIT, $0-56
    MOVQ x_base+0(FP), SI
    MOVQ x_len+8(FP), CX
    MOVQ y_base+24(FP), DI
    VXORPD Y0, Y0, Y0
    VXORPD Y1, Y1, Y1
    CMPQ CX, $8
    JL ddot_tail4

ddot_loop8:
    VMOVUPD (SI), Y2
    VMOVUPD 32(SI), Y3
    VFMADD231PD (DI), Y2, Y0
    VFMADD231PD 32(DI), Y3, Y1
    ADDQ $64, SI
    ADDQ $64, DI
    SUBQ $8, CX
    CMPQ CX, $8
    JGE ddot_loop8

ddot_tail4:
    VADDPD Y1, Y0, Y0
    CMPQ CX, $4
    JL ddot_reduce
    VMOVUPD (SI), Y2
    VFMADD231PD (DI), Y2, Y0
    ADDQ $32, SI
    ADDQ $32, DI
    SUBQ $4, CX

ddot_reduce:
    VEXTRACTF128 $1, Y0, X1
    VADDPD X1, X0, X0
    VHADDPD X0, X0, X0
    TESTQ CX, CX
    JZ ddot_done

ddot_scalar:
    VMOVSD (SI), X1
    VFMADD231SD (DI), X1, X0
    ADDQ $8, SI
    ADDQ $8, DI
    DECQ CX
    JNZ ddot_scalar

ddot_done:
    VMOVSD X0, ret+48(FP)
    VZEROUPPER
    RET
