// Conv1D k=3 stride=1 inner loop for amd64.
// func conv1dK3S1Inner(out, input *float32, w0, w1, w2 float32, n int)
//
// Accumulates: out[j] += w0*input[j-1] + w1*input[j] + w2*input[j+1]
// for j in 0..n-1 (boundary zero-pad).

#include "textflag.h"

TEXT ·conv1dK3S1Inner(SB), NOSPLIT, $0-40
    MOVQ out+0(FP), DI
    MOVQ input+8(FP), SI
    MOVSS w0+16(FP), X0
    MOVSS w1+20(FP), X1
    MOVSS w2+24(FP), X2
    MOVQ n+32(FP), CX

    CMPQ CX, $0
    JLE done

    // Handle j=0 (no left neighbor)
    MOVSS (SI), X3           // input[0]
    MULSS X1, X3             // w1*input[0]
    MOVSS (DI), X6
    ADDSS X3, X6
    CMPQ CX, $1
    JLE store_first
    MOVSS 4(SI), X3          // input[1]
    MULSS X2, X3             // w2*input[1]
    ADDSS X3, X6
store_first:
    MOVSS X6, (DI)

    CMPQ CX, $2
    JL done

    // Main loop j=1..n-2
    MOVQ $1, DX              // j = 1

    // Check if we can use AVX (need at least 8+2 elements from j=1)
    LEAQ -1(CX), R8          // R8 = n-1 (upper bound exclusive for middle)
    LEAQ -8(R8), R9           // R9 = n-9 (AVX upper bound)
    CMPQ DX, R9
    JGE scalar_middle

    // Broadcast weights for AVX
    VBROADCASTSS X0, Y0
    VBROADCASTSS X1, Y1
    VBROADCASTSS X2, Y2

avx_loop:
    // j = DX, load input[j-1..j+6], input[j..j+7], input[j+1..j+8]
    LEAQ (SI)(DX*4), AX
    VMOVUPS -4(AX), Y3       // input[j-1..j+6]
    VMOVUPS (AX), Y4         // input[j..j+7]
    VMOVUPS 4(AX), Y5        // input[j+1..j+8]

    // Load out[j..j+7]
    LEAQ (DI)(DX*4), BX
    VMOVUPS (BX), Y6

    // out += w0*left + w1*center + w2*right
    VFMADD231PS Y0, Y3, Y6
    VFMADD231PS Y1, Y4, Y6
    VFMADD231PS Y2, Y5, Y6

    VMOVUPS Y6, (BX)

    ADDQ $8, DX
    CMPQ DX, R9
    JL avx_loop

scalar_middle:
    // Scalar loop for remaining middle elements
    LEAQ -1(CX), R8          // n-1
middle_loop:
    CMPQ DX, R8
    JGE handle_last

    LEAQ (SI)(DX*4), AX
    LEAQ (DI)(DX*4), BX
    MOVSS (BX), X6           // out[j]

    MOVSS -4(AX), X3        // input[j-1]
    MULSS X0, X3
    ADDSS X3, X6

    MOVSS (AX), X3           // input[j]
    MULSS X1, X3
    ADDSS X3, X6

    MOVSS 4(AX), X3          // input[j+1]
    MULSS X2, X3
    ADDSS X3, X6

    MOVSS X6, (BX)
    INCQ DX
    JMP middle_loop

handle_last:
    // Handle j=n-1 (no right neighbor)
    LEAQ -1(CX), R8
    LEAQ (SI)(R8*4), AX
    LEAQ (DI)(R8*4), BX
    MOVSS (BX), X6

    MOVSS -4(AX), X3        // input[n-2]
    MULSS X0, X3
    ADDSS X3, X6

    MOVSS (AX), X3           // input[n-1]
    MULSS X1, X3
    ADDSS X3, X6

    MOVSS X6, (BX)

done:
    VZEROUPPER
    RET
