// Packed GGUF Q4_0 x Q8_0 AVX2/F16C row dot.
// The two four-lane accumulators and final reduction match llama.cpp's
// AVX2 reduction order and dotQ4_0Q8_0Scalar exactly.

#include "textflag.h"

// func dotQ4_0Q8_0AVX2(raw []byte, y []q8_0Block, blocks int) float32
TEXT ·dotQ4_0Q8_0AVX2(SB), NOSPLIT, $0-56
	MOVQ raw_base+0(FP), SI
	MOVQ y_base+24(FP), DI
	MOVQ blocks+48(FP), CX

	VXORPS X0, X0, X0              // low-nibble lanes 0..3
	VXORPS X1, X1, X1              // high-nibble lanes 0..3

	MOVL $0x00080008, AX
	VMOVD AX, X15
	VPBROADCASTD X15, Y15          // sixteen int16 values of 8
	VPCMPEQW Y14, Y14, Y14
	VPSRLW $12, Y14, Y14           // sixteen int16 values of 0x000f

	TESTQ CX, CX
	JZ reduce

loop:
	// Convert the block's fp16 scale and multiply by the Q8_0 fp32 scale.
	MOVW (SI), AX
	VMOVD AX, X12
	VCVTPH2PS X12, X12
	VMULSS (DI), X12, X12
	VSHUFPS $0, X12, X12, X12

	// Expand 16 packed bytes to int16 and form low-nibble values in [-8,7].
	VPMOVZXBW 2(SI), Y2
	VPAND Y14, Y2, Y3
	VPSUBW Y15, Y3, Y3
	VPMOVSXBW 4(DI), Y4
	VPMADDWD Y4, Y3, Y5            // eight adjacent-pair sums
	VPHADDD Y5, Y5, Y5             // duplicated four-product sums per 128-bit lane
	VEXTRACTI128 $1, Y5, X6
	VMOVLHPS X6, X5, X5            // [sum0,sum1,sum2,sum3]
	VCVTDQ2PS X5, X5
	VFMADD231PS X12, X5, X0

	// High nibbles correspond to the upper 16 Q8 activation values.
	VPSRLW $4, Y2, Y3
	VPAND Y14, Y3, Y3
	VPSUBW Y15, Y3, Y3
	VPMOVSXBW 20(DI), Y4
	VPMADDWD Y4, Y3, Y5
	VPHADDD Y5, Y5, Y5
	VEXTRACTI128 $1, Y5, X6
	VMOVLHPS X6, X5, X5
	VCVTDQ2PS X5, X5
	VFMADD231PS X12, X5, X1

	ADDQ $18, SI
	ADDQ $36, DI
	DECQ CX
	JNZ loop

reduce:
	VADDPS X1, X0, X0              // r0..r3
	VPSRLDQ $8, X0, X2
	VADDPS X2, X0, X0              // [r0+r2,r1+r3,...]
	VHADDPS X0, X0, X0
	VMOVSS X0, ret+56(FP)
	VZEROUPPER
	RET
