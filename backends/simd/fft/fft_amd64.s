// This file contains the Plan9 assembly for the AVX2 FFT butterfly kernel.
// It processes 4 butterfly pairs in parallel using 256-bit YMM registers.
//
// func fftButterfly4(re, im *float64, twRe, twIm *float64, aOff, bOff int, stride int)
//
// For i in 0..3 (stride=1, contiguous):
//   tr = twRe[i]*re[bOff+i] - twIm[i]*im[bOff+i]
//   ti = twRe[i]*im[bOff+i] + twIm[i]*re[bOff+i]
//   re[bOff+i] = re[aOff+i] - tr
//   im[bOff+i] = im[aOff+i] - ti
//   re[aOff+i] = re[aOff+i] + tr
//   im[aOff+i] = im[aOff+i] + ti

#include "textflag.h"

// func fftButterfly4(re, im *float64, twRe, twIm *float64, aOff, bOff int, stride int)
TEXT ·fftButterfly4(SB), NOSPLIT, $0-56
    // Load arguments
    MOVQ re+0(FP), AX       // AX = &re[0]
    MOVQ im+8(FP), BX       // BX = &im[0]
    MOVQ twRe+16(FP), CX    // CX = &twRe[0]
    MOVQ twIm+24(FP), DX    // DX = &twIm[0]
    MOVQ aOff+32(FP), SI    // SI = aOff
    MOVQ bOff+40(FP), DI    // DI = bOff

    // Convert offsets to byte offsets (float64 = 8 bytes)
    SHLQ $3, SI              // SI *= 8
    SHLQ $3, DI              // DI *= 8

    // Load 4 twiddle factors
    VMOVUPD (CX), Y0        // Y0 = twRe[0..3]
    VMOVUPD (DX), Y1        // Y1 = twIm[0..3]

    // Load re[b], im[b] (4 contiguous float64)
    VMOVUPD (AX)(DI*1), Y2  // Y2 = re[bOff..bOff+3]
    VMOVUPD (BX)(DI*1), Y3  // Y3 = im[bOff..bOff+3]

    // Load re[a], im[a]
    VMOVUPD (AX)(SI*1), Y4  // Y4 = re[aOff..aOff+3]
    VMOVUPD (BX)(SI*1), Y5  // Y5 = im[aOff..aOff+3]

    // tr = twRe * re[b] - twIm * im[b]
    VMULPD Y0, Y2, Y6       // Y6 = twRe * re[b]
    VMULPD Y1, Y3, Y7       // Y7 = twIm * im[b]
    VSUBPD Y7, Y6, Y6       // Y6 = tr

    // ti = twRe * im[b] + twIm * re[b]
    VMULPD Y0, Y3, Y7       // Y7 = twRe * im[b]
    VMULPD Y1, Y2, Y8       // Y8 = twIm * re[b]
    VADDPD Y8, Y7, Y7       // Y7 = ti

    // re[b] = re[a] - tr
    VSUBPD Y6, Y4, Y8       // Y8 = re[a] - tr
    VMOVUPD Y8, (AX)(DI*1)  // store re[b]

    // im[b] = im[a] - ti
    VSUBPD Y7, Y5, Y8       // Y8 = im[a] - ti
    VMOVUPD Y8, (BX)(DI*1)  // store im[b]

    // re[a] = re[a] + tr
    VADDPD Y6, Y4, Y4       // Y4 = re[a] + tr
    VMOVUPD Y4, (AX)(SI*1)  // store re[a]

    // im[a] = im[a] + ti
    VADDPD Y7, Y5, Y5       // Y5 = im[a] + ti
    VMOVUPD Y5, (BX)(SI*1)  // store im[a]

    VZEROUPPER
    RET
