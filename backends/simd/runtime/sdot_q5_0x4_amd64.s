#include "textflag.h"

// Shift-count vectors for the four 8-value chunks of a Q5_0 block, identical to
// sdotQ5_0Asm's Y8/Y10/Y11/Y12: chunk 0 shifts qh<<4 by 0..7, chunks 1..3
// shift qh by 4..11, 12..19 and 20..27, moving high bit i to bit 4.
DATA q5x4s<>+0(SB)/4, $0
DATA q5x4s<>+4(SB)/4, $1
DATA q5x4s<>+8(SB)/4, $2
DATA q5x4s<>+12(SB)/4, $3
DATA q5x4s<>+16(SB)/4, $4
DATA q5x4s<>+20(SB)/4, $5
DATA q5x4s<>+24(SB)/4, $6
DATA q5x4s<>+28(SB)/4, $7
DATA q5x4s<>+32(SB)/4, $4
DATA q5x4s<>+36(SB)/4, $5
DATA q5x4s<>+40(SB)/4, $6
DATA q5x4s<>+44(SB)/4, $7
DATA q5x4s<>+48(SB)/4, $8
DATA q5x4s<>+52(SB)/4, $9
DATA q5x4s<>+56(SB)/4, $10
DATA q5x4s<>+60(SB)/4, $11
DATA q5x4s<>+64(SB)/4, $12
DATA q5x4s<>+68(SB)/4, $13
DATA q5x4s<>+72(SB)/4, $14
DATA q5x4s<>+76(SB)/4, $15
DATA q5x4s<>+80(SB)/4, $16
DATA q5x4s<>+84(SB)/4, $17
DATA q5x4s<>+88(SB)/4, $18
DATA q5x4s<>+92(SB)/4, $19
DATA q5x4s<>+96(SB)/4, $20
DATA q5x4s<>+100(SB)/4, $21
DATA q5x4s<>+104(SB)/4, $22
DATA q5x4s<>+108(SB)/4, $23
DATA q5x4s<>+112(SB)/4, $24
DATA q5x4s<>+116(SB)/4, $25
DATA q5x4s<>+120(SB)/4, $26
DATA q5x4s<>+124(SB)/4, $27
GLOBL q5x4s<>(SB), RODATA|NOPTR, $128

// func sdotQ5_0x4Asm(x []float32, stride int, raw []byte, blocks int, out *[4]float32)
// Four input rows (x, x+stride, x+2*stride, x+3*stride floats) against one
// Q5_0 weight row. Each block's weight chunk is dequantised once with exactly
// sdotQ5_0Asm's instructions (single VMULPS rounding), then FMA'd into each
// row's own even/odd accumulator pair; each row's reduction is sdotQ5_0Asm's.
// Every out[r] is therefore bit-identical to sdotQ5_0Asm(row r, raw, blocks).
TEXT ·sdotQ5_0x4Asm(SB), NOSPLIT, $0-72
    MOVQ    x_base+0(FP), SI
    MOVQ    stride+24(FP), AX
    SHLQ    $2, AX                     // row stride in bytes
    LEAQ    (SI)(AX*1), R8
    LEAQ    (R8)(AX*1), R9
    LEAQ    (R9)(AX*1), R10
    MOVQ    raw_base+32(FP), DI
    MOVQ    blocks+56(FP), CX

    VXORPS  Y0, Y0, Y0                 // row 0 even
    VXORPS  Y1, Y1, Y1                 // row 0 odd
    VXORPS  Y6, Y6, Y6                 // row 1 even
    VXORPS  Y7, Y7, Y7                 // row 1 odd
    VXORPS  Y8, Y8, Y8                 // row 2 even
    VXORPS  Y11, Y11, Y11              // row 2 odd
    VXORPS  Y10, Y10, Y10              // row 3 even
    VXORPS  Y12, Y12, Y12              // row 3 odd

    VPCMPEQD Y14, Y14, Y14
    VPSRLD  $31, Y14, Y14              // 1
    VPSLLD  $4, Y14, Y13               // 16
    VPSUBD  Y14, Y13, Y15              // 15

q5x4_loop:
    VPBROADCASTW (DI), X9
    VCVTPH2PS X9, Y9                   // d in all lanes
    VPBROADCASTD 2(DI), Y3             // qh

    // chunk 0 -> even
    VPMOVZXBD 6(DI), Y5
    VPAND   Y15, Y5, Y2
    VPSLLD  $4, Y3, Y4
    VPSRLVD q5x4s<>+0(SB), Y4, Y4
    VPANDN  Y13, Y4, Y4
    VPSUBD  Y4, Y2, Y2
    VCVTDQ2PS Y2, Y2
    VMULPS  Y9, Y2, Y2
    VFMADD231PS (SI), Y2, Y0
    VFMADD231PS (R8), Y2, Y6
    VFMADD231PS (R9), Y2, Y8
    VFMADD231PS (R10), Y2, Y10

    // chunk 1 -> odd
    VPMOVZXBD 14(DI), Y5
    VPAND   Y15, Y5, Y2
    VPSRLVD q5x4s<>+32(SB), Y3, Y4
    VPANDN  Y13, Y4, Y4
    VPSUBD  Y4, Y2, Y2
    VCVTDQ2PS Y2, Y2
    VMULPS  Y9, Y2, Y2
    VFMADD231PS 32(SI), Y2, Y1
    VFMADD231PS 32(R8), Y2, Y7
    VFMADD231PS 32(R9), Y2, Y11
    VFMADD231PS 32(R10), Y2, Y12

    // chunk 2 -> even
    VPMOVZXBD 6(DI), Y5
    VPSRLD  $4, Y5, Y2
    VPSRLVD q5x4s<>+64(SB), Y3, Y4
    VPANDN  Y13, Y4, Y4
    VPSUBD  Y4, Y2, Y2
    VCVTDQ2PS Y2, Y2
    VMULPS  Y9, Y2, Y2
    VFMADD231PS 64(SI), Y2, Y0
    VFMADD231PS 64(R8), Y2, Y6
    VFMADD231PS 64(R9), Y2, Y8
    VFMADD231PS 64(R10), Y2, Y10

    // chunk 3 -> odd
    VPMOVZXBD 14(DI), Y5
    VPSRLD  $4, Y5, Y2
    VPSRLVD q5x4s<>+96(SB), Y3, Y4
    VPANDN  Y13, Y4, Y4
    VPSUBD  Y4, Y2, Y2
    VCVTDQ2PS Y2, Y2
    VMULPS  Y9, Y2, Y2
    VFMADD231PS 96(SI), Y2, Y1
    VFMADD231PS 96(R8), Y2, Y7
    VFMADD231PS 96(R9), Y2, Y11
    VFMADD231PS 96(R10), Y2, Y12

    ADDQ    $128, SI
    ADDQ    $128, R8
    ADDQ    $128, R9
    ADDQ    $128, R10
    ADDQ    $22, DI
    DECQ    CX
    JNZ     q5x4_loop

    MOVQ    out+64(FP), DX

    VADDPS  Y1, Y0, Y0
    VEXTRACTF128 $1, Y0, X1
    VADDPS  X1, X0, X0
    VHADDPS X0, X0, X0
    VHADDPS X0, X0, X0
    VMOVSS  X0, 0(DX)

    VADDPS  Y7, Y6, Y6
    VEXTRACTF128 $1, Y6, X7
    VADDPS  X7, X6, X6
    VHADDPS X6, X6, X6
    VHADDPS X6, X6, X6
    VMOVSS  X6, 4(DX)

    VADDPS  Y11, Y8, Y8
    VEXTRACTF128 $1, Y8, X11
    VADDPS  X11, X8, X8
    VHADDPS X8, X8, X8
    VHADDPS X8, X8, X8
    VMOVSS  X8, 8(DX)

    VADDPS  Y12, Y10, Y10
    VEXTRACTF128 $1, Y10, X12
    VADDPS  X12, X10, X10
    VHADDPS X10, X10, X10
    VHADDPS X10, X10, X10
    VMOVSS  X10, 12(DX)

    VZEROUPPER
    RET
