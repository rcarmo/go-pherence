//go:build arm64

#include "textflag.h"

// Four BF16 weight rows, one shared F32 activation stream. Each row retains
// the scalar reference's ascending-column fused F32 accumulation order. This avoids
// horizontal reductions and reads each activation once for four output rows.
// func bf16DotF32x4Arm64Asm(w []uint16, x []float32, cols int) (float32, float32, float32, float32)
TEXT ·bf16DotF32x4Arm64Asm(SB), NOSPLIT, $0-104
    MOVD w_base+0(FP), R0
    MOVD x_base+24(FP), R1
    MOVD cols+48(FP), R2
    MOVD R2, R3
    LSL $1, R3, R3
    ADD R0, R3, R4             // row 1 BF16
    ADD R4, R3, R5             // row 2 BF16
    ADD R5, R3, R6             // row 3 BF16
    MOVD $0, R7
    FMOVS R7, F0
    FMOVS R7, F1
    FMOVS R7, F2
    FMOVS R7, F3
loop:
    CBZ R2, done
    FMOVS (R1), F4
    MOVHU (R0), R7
    LSL $16, R7, R7
    FMOVS R7, F5
    FMADDS F5, F0, F4, F0
    MOVHU (R4), R7
    LSL $16, R7, R7
    FMOVS R7, F5
    FMADDS F5, F1, F4, F1
    MOVHU (R5), R7
    LSL $16, R7, R7
    FMOVS R7, F5
    FMADDS F5, F2, F4, F2
    MOVHU (R6), R7
    LSL $16, R7, R7
    FMOVS R7, F5
    FMADDS F5, F3, F4, F3
    ADD $2, R0
    ADD $2, R4
    ADD $2, R5
    ADD $2, R6
    ADD $4, R1
    SUB $1, R2, R2
    B loop
done:
    FMOVS F0, ret0+56(FP)
    FMOVS F1, ret1+60(FP)
    FMOVS F2, ret2+64(FP)
    FMOVS F3, ret3+68(FP)
    RET
