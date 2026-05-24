// NEON FFT butterfly for arm64.
// func fftButterfly2NEON(re, im *float64, twRe, twIm *float64, aOff, bOff int)
//
// Processes 2 float64 butterflies using NEON FMUL/FADD/FSUB (scalar pairs).
// Go Plan9 arm64 assembler does not have native float64x2 vector mnemonics,
// so we process 2 scalars sequentially with FMUL/FADD/FSUB instructions.

#include "textflag.h"

TEXT ·fftButterfly2NEON(SB), NOSPLIT, $0-48
    MOVD re+0(FP), R0
    MOVD im+8(FP), R1
    MOVD twRe+16(FP), R2
    MOVD twIm+24(FP), R3
    MOVD aOff+32(FP), R4
    MOVD bOff+40(FP), R5

    // Byte offsets
    LSL $3, R4, R4
    LSL $3, R5, R5

    // Process butterfly 0
    // Load twiddle
    FMOVD (R2), F0          // F0 = twRe[0]
    FMOVD (R3), F1          // F1 = twIm[0]
    // Load re[b], im[b]
    ADD R0, R5, R6
    FMOVD (R6), F2          // F2 = re[b+0]
    ADD R1, R5, R6
    FMOVD (R6), F3          // F3 = im[b+0]
    // Load re[a], im[a]
    ADD R0, R4, R6
    FMOVD (R6), F4          // F4 = re[a+0]
    ADD R1, R4, R6
    FMOVD (R6), F5          // F5 = im[a+0]

    // tr = twRe*re[b] - twIm*im[b]
    FMULD F0, F2, F6        // F6 = twRe * re[b]
    FMULD F1, F3, F7        // F7 = twIm * im[b]
    FSUBD F7, F6, F6        // F6 = tr

    // ti = twRe*im[b] + twIm*re[b]
    FMULD F0, F3, F7        // F7 = twRe * im[b]
    FMULD F1, F2, F8        // F8 = twIm * re[b]
    FADDD F8, F7, F7        // F7 = ti

    // Store results
    FSUBD F6, F4, F8        // re[b] = re[a] - tr
    ADD R0, R5, R6
    FMOVD F8, (R6)
    FSUBD F7, F5, F8        // im[b] = im[a] - ti
    ADD R1, R5, R6
    FMOVD F8, (R6)
    FADDD F6, F4, F4        // re[a] = re[a] + tr
    ADD R0, R4, R6
    FMOVD F4, (R6)
    FADDD F7, F5, F5        // im[a] = im[a] + ti
    ADD R1, R4, R6
    FMOVD F5, (R6)

    // Process butterfly 1 (offset +8 bytes)
    ADD $8, R2, R7
    ADD $8, R3, R8
    FMOVD (R7), F0          // twRe[1]
    FMOVD (R8), F1          // twIm[1]

    ADD $8, R5, R9          // b+1 byte offset
    ADD R0, R9, R6
    FMOVD (R6), F2          // re[b+1]
    ADD R1, R9, R6
    FMOVD (R6), F3          // im[b+1]

    ADD $8, R4, R10         // a+1 byte offset
    ADD R0, R10, R6
    FMOVD (R6), F4          // re[a+1]
    ADD R1, R10, R6
    FMOVD (R6), F5          // im[a+1]

    FMULD F0, F2, F6
    FMULD F1, F3, F7
    FSUBD F7, F6, F6        // tr

    FMULD F0, F3, F7
    FMULD F1, F2, F8
    FADDD F8, F7, F7        // ti

    FSUBD F6, F4, F8
    ADD R0, R9, R6
    FMOVD F8, (R6)
    FSUBD F7, F5, F8
    ADD R1, R9, R6
    FMOVD F8, (R6)
    FADDD F6, F4, F4
    ADD R0, R10, R6
    FMOVD F4, (R6)
    FADDD F7, F5, F5
    ADD R1, R10, R6
    FMOVD F5, (R6)

    RET
