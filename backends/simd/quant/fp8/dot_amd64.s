// dot_amd64.s — AVX2/FMA FP8 E4M3 LUT dot product.
//
// func dotE4M3AVX2(x []float32, w []byte, lut *[256]float32) float32
//
// Computes sum_j DecodeE4M3(w[j]) * x[j]. The caller applies the FP8 row
// scale and optional bias. We keep weights compressed and gather decoded
// float32 values from the 256-entry LUT eight bytes at a time.

#include "textflag.h"

TEXT ·dotE4M3AVX2(SB), NOSPLIT, $0-60
    MOVQ    x_base+0(FP), SI
    MOVQ    x_len+8(FP), CX
    MOVQ    w_base+24(FP), DI
    MOVQ    lut+48(FP), BX

    VXORPS  Y0, Y0, Y0          // vector accumulator

    CMPQ    CX, $8
    JL      reduce

loop8:
    // Convert 8 E4M3 bytes to 8 dword LUT indices.
    VPMOVZXBD (DI), Y1

    // Gather decoded E4M3 float32 values: Y2[i] = lut[index[i]].
    VPCMPEQD Y3, Y3, Y3        // all-ones gather mask; gather clears it
    VGATHERDPS Y3, (BX)(Y1*4), Y2

    // Accumulate decoded_w * x.
    VMOVUPS (SI), Y4
    VFMADD231PS Y4, Y2, Y0

    ADDQ    $32, SI            // 8 float32 activations
    ADDQ    $8, DI             // 8 FP8 weights
    SUBQ    $8, CX
    CMPQ    CX, $8
    JGE     loop8

reduce:
    // Horizontal reduce Y0 into scalar X0.
    VEXTRACTF128 $1, Y0, X1
    VADDPS  X1, X0, X0
    VHADDPS X0, X0, X0
    VHADDPS X0, X0, X0

    TESTQ   CX, CX
    JZ      done

tail:
    MOVBLZX (DI), DX
    VMOVSS  (BX)(DX*4), X1
    VMOVSS  (SI), X2
    VFMADD231SS X2, X1, X0
    ADDQ    $4, SI
    INCQ    DI
    DECQ    CX
    JNZ     tail

done:
    VMOVSS  X0, ret+56(FP)
    VZEROUPPER
    RET
