// Conv1D k=3 stride=1 inner loop for arm64.
// func conv1dK3S1InnerNEON(out, input *float32, w0, w1, w2 float32, n int)
//
// Accumulates: out[j] += w0*input[j-1] + w1*input[j] + w2*input[j+1]
// Uses scalar FP ops (avoids Go bounds checks in the hot loop).

#include "textflag.h"

TEXT ·conv1dK3S1InnerNEON(SB), NOSPLIT, $0-40
    MOVD out+0(FP), R0
    MOVD input+8(FP), R1
    FMOVS w0+16(FP), F0
    FMOVS w1+20(FP), F1
    FMOVS w2+24(FP), F2
    MOVD n+32(FP), R2

    CMP $0, R2
    BLE done

    // j=0: out[0] += w1*input[0] + w2*input[1]
    FMOVS (R1), F3
    FMULS F1, F3, F6
    CMP $1, R2
    BLE store_j0
    FMOVS 4(R1), F4
    FMULS F2, F4, F4
    FADDS F4, F6, F6
store_j0:
    FMOVS (R0), F7
    FADDS F6, F7, F7
    FMOVS F7, (R0)

    CMP $2, R2
    BLT done

    // Middle: j=1..n-2
    MOVD $1, R3
    SUB $1, R2, R4           // R4 = n-1

middle_loop:
    CMP R3, R4
    BLE handle_last

    LSL $2, R3, R5           // byte offset = j*4
    ADD R1, R5, R6           // &input[j]
    ADD R0, R5, R7           // &out[j]

    FMOVS (R7), F6           // out[j]

    FMOVS -4(R6), F3        // input[j-1]
    FMULS F0, F3, F3
    FADDS F3, F6, F6

    FMOVS (R6), F3           // input[j]
    FMULS F1, F3, F3
    FADDS F3, F6, F6

    FMOVS 4(R6), F3          // input[j+1]
    FMULS F2, F3, F3
    FADDS F3, F6, F6

    FMOVS F6, (R7)

    ADD $1, R3, R3
    B middle_loop

handle_last:
    // j=n-1: out[n-1] += w0*input[n-2] + w1*input[n-1]
    SUB $1, R2, R3
    LSL $2, R3, R5
    ADD R1, R5, R6
    ADD R0, R5, R7

    FMOVS (R7), F6
    FMOVS -4(R6), F3
    FMULS F0, F3, F3
    FADDS F3, F6, F6
    FMOVS (R6), F3
    FMULS F1, F3, F3
    FADDS F3, F6, F6
    FMOVS F6, (R7)

done:
    RET
