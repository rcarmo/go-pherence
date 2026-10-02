#include "textflag.h"

// Per-lane right shifts that move Q5_0 high bit i to bit 4: chunk 0 uses
// qh<<4 shifted by i (0..7); chunks 1..3 use qh shifted by i-4.
DATA q5hs<>+0(SB)/4, $0
DATA q5hs<>+4(SB)/4, $1
DATA q5hs<>+8(SB)/4, $2
DATA q5hs<>+12(SB)/4, $3
DATA q5hs<>+16(SB)/4, $4
DATA q5hs<>+20(SB)/4, $5
DATA q5hs<>+24(SB)/4, $6
DATA q5hs<>+28(SB)/4, $7
GLOBL q5hs<>(SB), RODATA|NOPTR, $32

// func sdotQ5_0Asm(x []float32, raw []byte, blocks int) float32
// Same accumulation as sdotAsm: element chunk k (8 floats) goes to Y0 when k is
// even and Y1 when odd; then Y0+Y1 and the identical horizontal reduction. Each
// weight is f16(d) * float32(q-16) with one VMULPS rounding (loader expression).
// q-16 is computed exactly
// as nib - (~(bit<<4) & 16), saving the shift/OR/bias uops of the direct form.
TEXT ·sdotQ5_0Asm(SB), NOSPLIT, $0-60
    MOVQ    x_base+0(FP), SI
    MOVQ    raw_base+24(FP), DI
    MOVQ    blocks+48(FP), CX

    VXORPS  Y0, Y0, Y0
    VXORPS  Y1, Y1, Y1

    VMOVDQU q5hs<>(SB), Y8             // 0..7 (on qh<<4)
    VPCMPEQD Y14, Y14, Y14
    VPSRLD  $31, Y14, Y14              // 1
    VPSLLD  $2, Y14, Y6                // 4
    VPSLLD  $3, Y14, Y7                // 8
    VPADDD  Y6, Y8, Y10                // 4..11 (on qh)
    VPADDD  Y7, Y10, Y11               // 12..19
    VPADDD  Y7, Y11, Y12               // 20..27
    VPSLLD  $4, Y14, Y13               // 16
    VPSUBD  Y14, Y13, Y15              // 15

q5_loop:
    VPBROADCASTW (DI), X9
    VCVTPH2PS X9, Y9                   // d in all lanes
    VPBROADCASTD 2(DI), Y3             // qh
    VPSLLD  $4, Y3, Y6                 // qh<<4 (bits 0..7 kept)
    VPMOVZXBD 6(DI), Y5                // qs[0..7]
    VPMOVZXBD 14(DI), Y7               // qs[8..15]

    // chunk 0 -> Y0
    VPAND   Y15, Y5, Y2
    VPSRLVD Y8, Y6, Y4
    VPANDN  Y13, Y4, Y4                // 16 - 16*bit
    VPSUBD  Y4, Y2, Y2
    VCVTDQ2PS Y2, Y2
    VMULPS  Y9, Y2, Y2
    VFMADD231PS (SI), Y2, Y0

    // chunk 1 -> Y1
    VPAND   Y15, Y7, Y2
    VPSRLVD Y10, Y3, Y4
    VPANDN  Y13, Y4, Y4
    VPSUBD  Y4, Y2, Y2
    VCVTDQ2PS Y2, Y2
    VMULPS  Y9, Y2, Y2
    VFMADD231PS 32(SI), Y2, Y1

    // chunk 2 -> Y0
    VPSRLD  $4, Y5, Y2
    VPSRLVD Y11, Y3, Y4
    VPANDN  Y13, Y4, Y4
    VPSUBD  Y4, Y2, Y2
    VCVTDQ2PS Y2, Y2
    VMULPS  Y9, Y2, Y2
    VFMADD231PS 64(SI), Y2, Y0

    // chunk 3 -> Y1
    VPSRLD  $4, Y7, Y2
    VPSRLVD Y12, Y3, Y4
    VPANDN  Y13, Y4, Y4
    VPSUBD  Y4, Y2, Y2
    VCVTDQ2PS Y2, Y2
    VMULPS  Y9, Y2, Y2
    VFMADD231PS 96(SI), Y2, Y1

    ADDQ    $128, SI
    ADDQ    $22, DI
    DECQ    CX
    JNZ     q5_loop

    VADDPS  Y1, Y0, Y0
    VEXTRACTF128 $1, Y0, X1
    VADDPS  X1, X0, X0
    VHADDPS X0, X0, X0
    VHADDPS X0, X0, X0
    VMOVSS  X0, ret+56(FP)
    VZEROUPPER
    RET
