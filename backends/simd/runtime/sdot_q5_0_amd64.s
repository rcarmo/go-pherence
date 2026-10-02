#include "textflag.h"

// Lane shift counts for the 32 Q5_0 high bits, one 8-lane vector per chunk.
DATA q5shift<>+0(SB)/4, $0
DATA q5shift<>+4(SB)/4, $1
DATA q5shift<>+8(SB)/4, $2
DATA q5shift<>+12(SB)/4, $3
DATA q5shift<>+16(SB)/4, $4
DATA q5shift<>+20(SB)/4, $5
DATA q5shift<>+24(SB)/4, $6
DATA q5shift<>+28(SB)/4, $7
GLOBL q5shift<>(SB), RODATA|NOPTR, $32

// func sdotQ5_0Asm(x []float32, raw []byte, blocks int) float32
// Same accumulation as sdotAsm: element chunk k (8 floats) goes to Y0 when k is
// even and Y1 when odd; then Y0+Y1 and the identical horizontal reduction. Each
// weight is f16(d) * float32(q-16) with one VMULPS rounding (loader expression).
TEXT ·sdotQ5_0Asm(SB), NOSPLIT, $0-60
    MOVQ    x_base+0(FP), SI
    MOVQ    raw_base+24(FP), DI
    MOVQ    blocks+48(FP), CX

    VXORPS  Y0, Y0, Y0
    VXORPS  Y1, Y1, Y1

    VMOVDQU q5shift<>(SB), Y8          // shifts 0..7
    VPCMPEQD Y14, Y14, Y14
    VPSRLD  $31, Y14, Y14              // 1
    VPSLLD  $3, Y14, Y7                // 8
    VPADDD  Y7, Y8, Y10                // 8..15
    VPADDD  Y7, Y10, Y11               // 16..23
    VPADDD  Y7, Y11, Y12               // 24..31
    VPSLLD  $4, Y14, Y13               // 16
    VPSUBD  Y14, Y13, Y15              // 15

q5_loop:
    MOVWLZX (DI), AX
    VMOVD   AX, X9
    VCVTPH2PS X9, X9
    VBROADCASTSS X9, Y9                // d
    VPBROADCASTD 2(DI), Y3             // qh

    // chunk 0: low nibbles of qs[0..7], bits 0..7 -> Y0
    VPMOVZXBD 6(DI), Y2
    VPAND   Y15, Y2, Y2
    VPSRLVD Y8, Y3, Y4
    VPAND   Y14, Y4, Y4
    VPSLLD  $4, Y4, Y4
    VPOR    Y4, Y2, Y2
    VPSUBD  Y13, Y2, Y2
    VCVTDQ2PS Y2, Y2
    VMULPS  Y9, Y2, Y2
    VFMADD231PS (SI), Y2, Y0

    // chunk 1: low nibbles of qs[8..15], bits 8..15 -> Y1
    VPMOVZXBD 14(DI), Y5
    VPAND   Y15, Y5, Y2
    VPSRLVD Y10, Y3, Y4
    VPAND   Y14, Y4, Y4
    VPSLLD  $4, Y4, Y4
    VPOR    Y4, Y2, Y2
    VPSUBD  Y13, Y2, Y2
    VCVTDQ2PS Y2, Y2
    VMULPS  Y9, Y2, Y2
    VFMADD231PS 32(SI), Y2, Y1

    // chunk 2: high nibbles of qs[0..7], bits 16..23 -> Y0
    VPMOVZXBD 6(DI), Y2
    VPSRLD  $4, Y2, Y2
    VPSRLVD Y11, Y3, Y4
    VPAND   Y14, Y4, Y4
    VPSLLD  $4, Y4, Y4
    VPOR    Y4, Y2, Y2
    VPSUBD  Y13, Y2, Y2
    VCVTDQ2PS Y2, Y2
    VMULPS  Y9, Y2, Y2
    VFMADD231PS 64(SI), Y2, Y0

    // chunk 3: high nibbles of qs[8..15], bits 24..31 -> Y1
    VPSRLD  $4, Y5, Y2
    VPSRLVD Y12, Y3, Y4
    VPAND   Y14, Y4, Y4
    VPSLLD  $4, Y4, Y4
    VPOR    Y4, Y2, Y2
    VPSUBD  Y13, Y2, Y2
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
