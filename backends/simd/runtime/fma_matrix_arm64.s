//go:build arm64

#include "textflag.h"

// Overwrite C in one pass with the existing serial-K SgemmNN tile order.
// Inputs were validated in FMAMatrixF32Checked; alpha is exactly 1.
TEXT ·fmaMatrixArm64Asm(SB), NOSPLIT, $0-96
    MOVD    m+72(FP), R0
    MOVD    n+80(FP), R1
    MOVD    k+88(FP), R2
    MOVD    a_base+24(FP), R3
    MOVD    b_base+48(FP), R4
    MOVD    dst_base+0(FP), R5
    MOVD    k+88(FP), R6
    MOVD    n+80(FP), R7
    MOVD    n+80(FP), R8

    LSL     $2, R6, R6
    LSL     $2, R7, R7
    LSL     $2, R8, R8

    CBZ     R0, nn_done

    MOVD    R3, R9               // A row ptr
    MOVD    R5, R10              // C row ptr

nn_i_loop:
    MOVD    R1, R11              // remaining columns
    MOVD    R10, R13             // &C[i, jj]
    MOVD    R4, R12              // &B[0, jj]

nn_tile16:
    CMP     $16, R11
    BLT     nn_tile4

    // +0 accumulators; no destination read or separate clear pass.
    VEOR    V0.B16, V0.B16, V0.B16
    VEOR    V1.B16, V1.B16, V1.B16
    VEOR    V2.B16, V2.B16, V2.B16
    VEOR    V3.B16, V3.B16, V3.B16

    MOVD    R9, R14              // &A[i, 0]
    MOVD    R12, R15             // &B[0, jj]
    MOVD    R2, R16              // k counter

nn_p16:
    // Broadcast A[i,p]; alpha is fixed at 1.
    FMOVS   (R14), F8
    VDUP    V8.S[0], V8.S4

    // Serial-K FMA into the output tile.
    VLD1    (R15), [V4.S4, V5.S4, V6.S4, V7.S4]
    VFMLA   V8.S4, V4.S4, V0.S4
    VFMLA   V8.S4, V5.S4, V1.S4
    VFMLA   V8.S4, V6.S4, V2.S4
    VFMLA   V8.S4, V7.S4, V3.S4

    ADD     $4, R14
    ADD     R7, R15
    SUB     $1, R16, R16
    CBNZ    R16, nn_p16

    // Store C tile
    VST1    [V0.S4, V1.S4, V2.S4, V3.S4], (R13)

    ADD     $64, R13
    ADD     $64, R12
    SUB     $16, R11, R11
    B       nn_tile16

nn_tile4:
    CMP     $4, R11
    BLT     nn_tile1

    VEOR    V0.B16, V0.B16, V0.B16

    MOVD    R9, R14
    MOVD    R12, R15
    MOVD    R2, R16

nn_p4:
    FMOVS   (R14), F8
    VDUP    V8.S[0], V8.S4
    VLD1    (R15), [V4.S4]
    VFMLA   V8.S4, V4.S4, V0.S4
    ADD     $4, R14
    ADD     R7, R15
    SUB     $1, R16, R16
    CBNZ    R16, nn_p4

    VST1    [V0.S4], (R13)

    ADD     $16, R13
    ADD     $16, R12
    SUB     $4, R11, R11
    B       nn_tile4

nn_tile1:
    CBZ     R11, nn_next_i

nn_tile1_loop:
    MOVD    $0, R17
    FMOVS   R17, F0

    MOVD    R9, R14
    MOVD    R12, R15
    MOVD    R2, R16

nn_p1:
    FMOVS   (R14), F4
    FMOVS   (R15), F5
    FMADDS  F4, F0, F5, F0
    ADD     $4, R14
    ADD     R7, R15
    SUB     $1, R16, R16
    CBNZ    R16, nn_p1

    FMOVS   F0, (R13)
    ADD     $4, R13
    ADD     $4, R12
    SUB     $1, R11, R11
    CBNZ    R11, nn_tile1_loop

nn_next_i:
    ADD     R6, R9
    ADD     R8, R10
    SUB     $1, R0, R0
    CBNZ    R0, nn_i_loop

nn_done:
    RET
