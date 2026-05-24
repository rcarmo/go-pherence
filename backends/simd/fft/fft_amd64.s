// This file contains the Plan9 assembly for the AVX2 FFT butterfly kernel.
// It processes 4 butterfly pairs in parallel using 256-bit YMM registers.
//
// func fftButterfly4(re, im *float64, twRe, twIm *float64, aOff, bOff int, stride int)
//
// Computes for i in 0..3:
//   tr = twRe[i]*re[bOff+i*stride] - twIm[i]*im[bOff+i*stride]
//   ti = twRe[i]*im[bOff+i*stride] + twIm[i]*re[bOff+i*stride]
//   re[bOff+i*stride] = re[aOff+i*stride] - tr
//   im[bOff+i*stride] = im[aOff+i*stride] - ti
//   re[aOff+i*stride] = re[aOff+i*stride] + tr
//   im[aOff+i*stride] = im[aOff+i*stride] + ti

#include "textflag.h"

// For now this is a stub — the Go fallback in fft_asm_amd64.go handles dispatch.
// The assembly body will use VMOVUPD, VMULPD, VFMADD/VFMSUB for the butterfly.
TEXT ·fftButterfly4(SB), NOSPLIT, $0-56
    RET
