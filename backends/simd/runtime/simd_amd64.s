// simd_amd64.s — AVX2/FMA SIMD kernels for Go (plan9 assembly)

#include "textflag.h"

// func Sdot(x, y []float32) float32
TEXT ·sdotAsm(SB), NOSPLIT, $0-52
    MOVQ    x_base+0(FP), SI
    MOVQ    x_len+8(FP), CX
    MOVQ    y_base+24(FP), DI

    VXORPS  Y0, Y0, Y0
    VXORPS  Y1, Y1, Y1

    CMPQ    CX, $16
    JL      sdot_post16

sdot_loop16:
    VMOVUPS (SI), Y2
    VMOVUPS 32(SI), Y3
    VFMADD231PS (DI), Y2, Y0
    VFMADD231PS 32(DI), Y3, Y1
    ADDQ    $64, SI
    ADDQ    $64, DI
    SUBQ    $16, CX
    CMPQ    CX, $16
    JGE     sdot_loop16

sdot_post16:
    VADDPS  Y1, Y0, Y0

    CMPQ    CX, $8
    JL      sdot_post8
    VMOVUPS (SI), Y2
    VFMADD231PS (DI), Y2, Y0
    ADDQ    $32, SI
    ADDQ    $32, DI
    SUBQ    $8, CX

sdot_post8:
    // Horizontal reduce Y0 → scalar in X0
    VEXTRACTF128 $1, Y0, X1
    VADDPS  X1, X0, X0
    VHADDPS X0, X0, X0
    VHADDPS X0, X0, X0

    // 4-wide tail (into X4, then add to X0)
    CMPQ    CX, $4
    JL      sdot_scalar_check
    VXORPS  X4, X4, X4
    VMOVUPS (SI), X2
    VMOVUPS (DI), X3
    VFMADD231PS X3, X2, X4
    // hsum X4
    VHADDPS X4, X4, X4
    VHADDPS X4, X4, X4
    VADDSS  X4, X0, X0
    ADDQ    $16, SI
    ADDQ    $16, DI
    SUBQ    $4, CX

sdot_scalar_check:
    TESTQ   CX, CX
    JZ      sdot_done

sdot_scalar:
    VMOVSS  (SI), X1
    VMOVSS  (DI), X2
    VFMADD231SS X2, X1, X0
    ADDQ    $4, SI
    ADDQ    $4, DI
    DECQ    CX
    JNZ     sdot_scalar

sdot_done:
    VMOVSS  X0, ret+48(FP)
    VZEROUPPER
    RET

// func Saxpy(alpha float32, x []float32, y []float32)
TEXT ·saxpyAsm(SB), NOSPLIT, $0-56
    MOVSS       alpha+0(FP), X8
    VBROADCASTSS X8, Y8
    MOVQ    x_base+8(FP), SI
    MOVQ    x_len+16(FP), CX
    MOVQ    y_base+32(FP), DI

    CMPQ    CX, $16
    JL      saxpy_post16

saxpy_loop16:
    VMOVUPS (DI), Y0
    VMOVUPS 32(DI), Y1
    VMOVUPS (SI), Y2
    VMOVUPS 32(SI), Y3
    VFMADD231PS Y8, Y2, Y0
    VFMADD231PS Y8, Y3, Y1
    VMOVUPS Y0, (DI)
    VMOVUPS Y1, 32(DI)
    ADDQ    $64, SI
    ADDQ    $64, DI
    SUBQ    $16, CX
    CMPQ    CX, $16
    JGE     saxpy_loop16

saxpy_post16:
    CMPQ    CX, $8
    JL      saxpy_post8
    VMOVUPS (DI), Y0
    VMOVUPS (SI), Y2
    VFMADD231PS Y8, Y2, Y0
    VMOVUPS Y0, (DI)
    ADDQ    $32, SI
    ADDQ    $32, DI
    SUBQ    $8, CX

saxpy_post8:
    CMPQ    CX, $4
    JL      saxpy_scalar_check
    VMOVUPS (DI), X0
    VMOVUPS (SI), X2
    VFMADD231PS X8, X2, X0
    VMOVUPS X0, (DI)
    ADDQ    $16, SI
    ADDQ    $16, DI
    SUBQ    $4, CX

saxpy_scalar_check:
    TESTQ   CX, CX
    JZ      saxpy_done

saxpy_scalar:
    VMOVSS  (DI), X0
    VMOVSS  (SI), X2
    VFMADD231SS X8, X2, X0
    VMOVSS  X0, (DI)
    ADDQ    $4, SI
    ADDQ    $4, DI
    DECQ    CX
    JNZ     saxpy_scalar

saxpy_done:
    VZEROUPPER
    RET

// func sdotx4Asm(w []float32, x []float32, stride int) (dot0,dot1,dot2,dot3 float32)
// Computes dot(w, x[row]) for four contiguous rows separated by stride.
// Requires len(w)%16==0; Go wrapper validates shape and falls back otherwise.
TEXT ·sdotx4Asm(SB), NOSPLIT, $0-72
    MOVQ    w_base+0(FP), SI
    MOVQ    w_len+8(FP), CX
    MOVQ    x_base+24(FP), DI
    MOVQ    stride+48(FP), R8
    SHLQ    $2, R8

    LEAQ    (DI)(R8*1), BX
    LEAQ    (BX)(R8*1), DX
    LEAQ    (DX)(R8*1), R9

    VXORPS  Y0, Y0, Y0      // row0 acc a
    VXORPS  Y1, Y1, Y1      // row0 acc b
    VXORPS  Y2, Y2, Y2      // row1 acc a
    VXORPS  Y3, Y3, Y3      // row1 acc b
    VXORPS  Y4, Y4, Y4      // row2 acc a
    VXORPS  Y5, Y5, Y5      // row2 acc b
    VXORPS  Y6, Y6, Y6      // row3 acc a
    VXORPS  Y7, Y7, Y7      // row3 acc b

    TESTQ   CX, CX
    JZ      sdotx4_reduce

sdotx4_loop16:
    VMOVUPS (SI), Y8
    VMOVUPS 32(SI), Y9

    VFMADD231PS (DI), Y8, Y0
    VFMADD231PS 32(DI), Y9, Y1

    VFMADD231PS (BX), Y8, Y2
    VFMADD231PS 32(BX), Y9, Y3

    VFMADD231PS (DX), Y8, Y4
    VFMADD231PS 32(DX), Y9, Y5

    VFMADD231PS (R9), Y8, Y6
    VFMADD231PS 32(R9), Y9, Y7

    ADDQ    $64, SI
    ADDQ    $64, DI
    ADDQ    $64, BX
    ADDQ    $64, DX
    ADDQ    $64, R9
    SUBQ    $16, CX
    JNZ     sdotx4_loop16

sdotx4_reduce:
    VADDPS  Y1, Y0, Y0
    VEXTRACTF128 $1, Y0, X10
    VADDPS  X10, X0, X0
    VHADDPS X0, X0, X0
    VHADDPS X0, X0, X0

    VADDPS  Y3, Y2, Y2
    VEXTRACTF128 $1, Y2, X10
    VADDPS  X10, X2, X2
    VHADDPS X2, X2, X2
    VHADDPS X2, X2, X2

    VADDPS  Y5, Y4, Y4
    VEXTRACTF128 $1, Y4, X10
    VADDPS  X10, X4, X4
    VHADDPS X4, X4, X4
    VHADDPS X4, X4, X4

    VADDPS  Y7, Y6, Y6
    VEXTRACTF128 $1, Y6, X10
    VADDPS  X10, X6, X6
    VHADDPS X6, X6, X6
    VHADDPS X6, X6, X6

    VMOVSS  X0, dot0+56(FP)
    VMOVSS  X2, dot1+60(FP)
    VMOVSS  X4, dot2+64(FP)
    VMOVSS  X6, dot3+68(FP)
    VZEROUPPER
    RET

// func dotRowsx4Asm(w []float32, x []float32, cols int) (dot0,dot1,dot2,dot3 float32)
// Computes one shared F32 activation vector against four consecutive F32 weight rows.
TEXT ·dotRowsx4Asm(SB), NOSPLIT, $0-72
    MOVQ    w_base+0(FP), SI
    MOVQ    x_base+24(FP), DI
    MOVQ    cols+48(FP), CX

    MOVQ    CX, R8
    SHLQ    $2, R8
    LEAQ    (SI)(R8*1), BX
    LEAQ    (BX)(R8*1), DX
    LEAQ    (DX)(R8*1), R9

    VXORPS  Y0, Y0, Y0
    VXORPS  Y1, Y1, Y1
    VXORPS  Y2, Y2, Y2
    VXORPS  Y3, Y3, Y3

    CMPQ    CX, $16
    JL      dotrowsx4_post16

dotrowsx4_loop16:
    VMOVUPS (DI), Y4
    VMOVUPS 32(DI), Y5

    VMOVUPS (SI), Y6
    VFMADD231PS Y4, Y6, Y0
    VMOVUPS 32(SI), Y7
    VFMADD231PS Y5, Y7, Y0

    VMOVUPS (BX), Y6
    VFMADD231PS Y4, Y6, Y1
    VMOVUPS 32(BX), Y7
    VFMADD231PS Y5, Y7, Y1

    VMOVUPS (DX), Y6
    VFMADD231PS Y4, Y6, Y2
    VMOVUPS 32(DX), Y7
    VFMADD231PS Y5, Y7, Y2

    VMOVUPS (R9), Y6
    VFMADD231PS Y4, Y6, Y3
    VMOVUPS 32(R9), Y7
    VFMADD231PS Y5, Y7, Y3

    ADDQ    $64, SI
    ADDQ    $64, BX
    ADDQ    $64, DX
    ADDQ    $64, R9
    ADDQ    $64, DI
    SUBQ    $16, CX
    CMPQ    CX, $16
    JGE     dotrowsx4_loop16

dotrowsx4_post16:
    CMPQ    CX, $8
    JL      dotrowsx4_reduce
    VMOVUPS (DI), Y4

    VMOVUPS (SI), Y6
    VFMADD231PS Y4, Y6, Y0
    VMOVUPS (BX), Y6
    VFMADD231PS Y4, Y6, Y1
    VMOVUPS (DX), Y6
    VFMADD231PS Y4, Y6, Y2
    VMOVUPS (R9), Y6
    VFMADD231PS Y4, Y6, Y3

    ADDQ    $32, SI
    ADDQ    $32, BX
    ADDQ    $32, DX
    ADDQ    $32, R9
    ADDQ    $32, DI
    SUBQ    $8, CX

dotrowsx4_reduce:
    VEXTRACTF128 $1, Y0, X8
    VADDPS  X8, X0, X0
    VHADDPS X0, X0, X0
    VHADDPS X0, X0, X0

    VEXTRACTF128 $1, Y1, X8
    VADDPS  X8, X1, X1
    VHADDPS X1, X1, X1
    VHADDPS X1, X1, X1

    VEXTRACTF128 $1, Y2, X8
    VADDPS  X8, X2, X2
    VHADDPS X2, X2, X2
    VHADDPS X2, X2, X2

    VEXTRACTF128 $1, Y3, X8
    VADDPS  X8, X3, X3
    VHADDPS X3, X3, X3
    VHADDPS X3, X3, X3

dotrowsx4_scalar_check:
    TESTQ   CX, CX
    JZ      dotrowsx4_done

dotrowsx4_scalar:
    VMOVSS  (DI), X4

    VMOVSS  (SI), X5
    VFMADD231SS X4, X5, X0
    VMOVSS  (BX), X5
    VFMADD231SS X4, X5, X1
    VMOVSS  (DX), X5
    VFMADD231SS X4, X5, X2
    VMOVSS  (R9), X5
    VFMADD231SS X4, X5, X3

    ADDQ    $4, SI
    ADDQ    $4, BX
    ADDQ    $4, DX
    ADDQ    $4, R9
    ADDQ    $4, DI
    DECQ    CX
    JNZ     dotrowsx4_scalar

dotrowsx4_done:
    VMOVSS  X0, dot0+56(FP)
    VMOVSS  X1, dot1+60(FP)
    VMOVSS  X2, dot2+64(FP)
    VMOVSS  X3, dot3+68(FP)
    VZEROUPPER
    RET

// func bf16DotF32x4Asm(w []uint16, x []float32, cols int) (dot0,dot1,dot2,dot3 float32)
// Computes four consecutive BF16 weight rows against one shared F32 activation.
TEXT ·bf16DotF32x4Asm(SB), NOSPLIT, $0-72
    MOVQ    w_base+0(FP), SI
    MOVQ    x_base+24(FP), DI
    MOVQ    cols+48(FP), CX

    MOVQ    CX, R8
    SHLQ    $1, R8
    LEAQ    (SI)(R8*1), BX
    LEAQ    (BX)(R8*1), DX
    LEAQ    (DX)(R8*1), R9

    VXORPS  Y0, Y0, Y0
    VXORPS  Y1, Y1, Y1
    VXORPS  Y2, Y2, Y2
    VXORPS  Y3, Y3, Y3

    CMPQ    CX, $8
    JL      bf16dotf32x4_reduce

bf16dotf32x4_loop8:
    VMOVUPS (DI), Y4

    VPMOVZXWD (SI), Y5
    VPSLLD  $16, Y5, Y5
    VPMOVZXWD (BX), Y6
    VPSLLD  $16, Y6, Y6
    VPMOVZXWD (DX), Y7
    VPSLLD  $16, Y7, Y7
    VPMOVZXWD (R9), Y8
    VPSLLD  $16, Y8, Y8

    VFMADD231PS Y4, Y5, Y0
    VFMADD231PS Y4, Y6, Y1
    VFMADD231PS Y4, Y7, Y2
    VFMADD231PS Y4, Y8, Y3

    ADDQ    $16, SI
    ADDQ    $16, BX
    ADDQ    $16, DX
    ADDQ    $16, R9
    ADDQ    $32, DI
    SUBQ    $8, CX
    CMPQ    CX, $8
    JGE     bf16dotf32x4_loop8

bf16dotf32x4_reduce:
    VEXTRACTF128 $1, Y0, X8
    VADDPS  X8, X0, X0
    VHADDPS X0, X0, X0
    VHADDPS X0, X0, X0

    VEXTRACTF128 $1, Y1, X9
    VADDPS  X9, X1, X1
    VHADDPS X1, X1, X1
    VHADDPS X1, X1, X1

    VEXTRACTF128 $1, Y2, X10
    VADDPS  X10, X2, X2
    VHADDPS X2, X2, X2
    VHADDPS X2, X2, X2

    VEXTRACTF128 $1, Y3, X11
    VADDPS  X11, X3, X3
    VHADDPS X3, X3, X3
    VHADDPS X3, X3, X3

    TESTQ   CX, CX
    JZ      bf16dotf32x4_done

bf16dotf32x4_scalar:
    VMOVSS  (DI), X4

    MOVWLZX (SI), R10
    SHLL    $16, R10
    MOVL    R10, X5
    VFMADD231SS X4, X5, X0

    MOVWLZX (BX), R11
    SHLL    $16, R11
    MOVL    R11, X5
    VFMADD231SS X4, X5, X1

    MOVWLZX (DX), R12
    SHLL    $16, R12
    MOVL    R12, X5
    VFMADD231SS X4, X5, X2

    MOVWLZX (R9), R13
    SHLL    $16, R13
    MOVL    R13, X5
    VFMADD231SS X4, X5, X3

    ADDQ    $2, SI
    ADDQ    $2, BX
    ADDQ    $2, DX
    ADDQ    $2, R9
    ADDQ    $4, DI
    DECQ    CX
    JNZ     bf16dotf32x4_scalar

bf16dotf32x4_done:
    VMOVSS  X0, dot0+56(FP)
    VMOVSS  X1, dot1+60(FP)
    VMOVSS  X2, dot2+64(FP)
    VMOVSS  X3, dot3+68(FP)
    VZEROUPPER
    RET

// func bf16DotBF16x4Asm(w []uint16, x []uint16, cols int) (dot0,dot1,dot2,dot3 float32)
// Computes four consecutive BF16 weight rows against one shared BF16 activation.
TEXT ·bf16DotBF16x4Asm(SB), NOSPLIT, $0-72
    MOVQ    w_base+0(FP), SI
    MOVQ    x_base+24(FP), DI
    MOVQ    cols+48(FP), CX

    MOVQ    CX, R8
    SHLQ    $1, R8
    LEAQ    (SI)(R8*1), BX
    LEAQ    (BX)(R8*1), DX
    LEAQ    (DX)(R8*1), R9

    VXORPS  Y0, Y0, Y0
    VXORPS  Y1, Y1, Y1
    VXORPS  Y2, Y2, Y2
    VXORPS  Y3, Y3, Y3

    CMPQ    CX, $8
    JL      bf16dotbf16x4_reduce

bf16dotbf16x4_loop8:
    VPMOVZXWD (DI), Y4
    VPSLLD  $16, Y4, Y4

    VPMOVZXWD (SI), Y5
    VPSLLD  $16, Y5, Y5
    VPMOVZXWD (BX), Y6
    VPSLLD  $16, Y6, Y6
    VPMOVZXWD (DX), Y7
    VPSLLD  $16, Y7, Y7
    VPMOVZXWD (R9), Y8
    VPSLLD  $16, Y8, Y8

    VFMADD231PS Y4, Y5, Y0
    VFMADD231PS Y4, Y6, Y1
    VFMADD231PS Y4, Y7, Y2
    VFMADD231PS Y4, Y8, Y3

    ADDQ    $16, SI
    ADDQ    $16, BX
    ADDQ    $16, DX
    ADDQ    $16, R9
    ADDQ    $16, DI
    SUBQ    $8, CX
    CMPQ    CX, $8
    JGE     bf16dotbf16x4_loop8

bf16dotbf16x4_reduce:
    VEXTRACTF128 $1, Y0, X8
    VADDPS  X8, X0, X0
    VHADDPS X0, X0, X0
    VHADDPS X0, X0, X0

    VEXTRACTF128 $1, Y1, X9
    VADDPS  X9, X1, X1
    VHADDPS X1, X1, X1
    VHADDPS X1, X1, X1

    VEXTRACTF128 $1, Y2, X10
    VADDPS  X10, X2, X2
    VHADDPS X2, X2, X2
    VHADDPS X2, X2, X2

    VEXTRACTF128 $1, Y3, X11
    VADDPS  X11, X3, X3
    VHADDPS X3, X3, X3
    VHADDPS X3, X3, X3

    TESTQ   CX, CX
    JZ      bf16dotbf16x4_done

bf16dotbf16x4_scalar:
    MOVWLZX (DI), R10
    SHLL    $16, R10
    MOVL    R10, X4

    MOVWLZX (SI), R11
    SHLL    $16, R11
    MOVL    R11, X5
    VFMADD231SS X4, X5, X0

    MOVWLZX (BX), R12
    SHLL    $16, R12
    MOVL    R12, X5
    VFMADD231SS X4, X5, X1

    MOVWLZX (DX), R13
    SHLL    $16, R13
    MOVL    R13, X5
    VFMADD231SS X4, X5, X2

    MOVWLZX (R9), R14
    SHLL    $16, R14
    MOVL    R14, X5
    VFMADD231SS X4, X5, X3

    ADDQ    $2, SI
    ADDQ    $2, BX
    ADDQ    $2, DX
    ADDQ    $2, R9
    ADDQ    $2, DI
    DECQ    CX
    JNZ     bf16dotbf16x4_scalar

bf16dotbf16x4_done:
    VMOVSS  X0, dot0+56(FP)
    VMOVSS  X1, dot1+60(FP)
    VMOVSS  X2, dot2+64(FP)
    VMOVSS  X3, dot3+68(FP)
    VZEROUPPER
    RET
