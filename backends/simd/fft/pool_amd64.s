// SIMD statistics pooling inner loop for amd64.
// func meanStdReduce(out_mean, out_std, input *float32, length int)
//
// Computes mean and standard deviation of input[0..length-1].
// Uses AVX2 for parallel accumulation.

#include "textflag.h"

// func meanStdReduce(out_mean, out_std, input *float32, length int)
TEXT ·meanStdReduce(SB), NOSPLIT, $0-32
    MOVQ out_mean+0(FP), DI  // &out_mean
    MOVQ out_std+8(FP), SI   // &out_std
    MOVQ input+16(FP), DX    // &input[0]
    MOVQ length+24(FP), CX   // length

    CMPQ CX, $0
    JLE done

    // Accumulate sum and sum-of-squares using AVX2
    VXORPS Y0, Y0, Y0       // Y0 = sum accumulator (8 floats)
    VXORPS Y1, Y1, Y1       // Y1 = sq accumulator (8 floats)

    MOVQ CX, AX
    SHRQ $3, AX             // AX = length / 8 (AVX iterations)
    CMPQ AX, $0
    JE scalar_accum

    XORQ BX, BX             // loop counter
avx_loop:
    VMOVUPS (DX)(BX*4), Y2  // Load 8 floats
    VADDPS Y2, Y0, Y0       // sum += x
    VFMADD231PS Y2, Y2, Y1  // sq += x*x
    ADDQ $8, BX
    DECQ AX
    JNZ avx_loop

    // Horizontal reduce Y0 (sum) and Y1 (sq) to scalars
    VEXTRACTF128 $1, Y0, X3
    VEXTRACTF128 $1, Y1, X4
    ADDPS X3, X0             // X0 = sum low + high
    ADDPS X4, X1             // X1 = sq low + high
    MOVHLPS X0, X3
    MOVHLPS X1, X4
    ADDPS X3, X0
    ADDPS X4, X1
    PSHUFD $0x55, X0, X3    // shuffle element 1
    PSHUFD $0x55, X1, X4
    ADDSS X3, X0
    ADDSS X4, X1

scalar_accum:
    // Handle remaining elements (length % 8)
    MOVQ CX, AX
    ANDQ $7, AX             // remainder
    CMPQ AX, $0
    JE compute_stats

    MOVQ CX, BX
    SUBQ AX, BX             // start offset for tail
tail_loop:
    MOVSS (DX)(BX*4), X2
    ADDSS X2, X0             // sum += x
    MULSS X2, X2
    ADDSS X2, X1             // sq += x*x
    INCQ BX
    DECQ AX
    JNZ tail_loop

compute_stats:
    // mean = sum / n
    CVTSL2SS CX, X2         // X2 = float(n)
    DIVSS X2, X0             // X0 = mean
    MOVSS X0, (DI)           // store mean

    // std = sqrt(sq/n - mean^2)
    DIVSS X2, X1             // X1 = sq/n
    MULSS X0, X0             // X0 = mean^2
    SUBSS X0, X1             // X1 = variance
    // Clamp negative variance to 0
    XORPS X3, X3
    MAXSS X3, X1
    SQRTSS X1, X1            // X1 = std
    MOVSS X1, (SI)           // store std

done:
    VZEROUPPER
    RET
