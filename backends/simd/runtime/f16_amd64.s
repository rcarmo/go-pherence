#include "textflag.h"

DATA ·f16AbsMask<>+0(SB)/8, $0x7fff7fff7fff7fff
DATA ·f16AbsMask<>+8(SB)/8, $0x7fff7fff7fff7fff
GLOBL ·f16AbsMask<>(SB), RODATA|NOPTR, $16

DATA ·f16NaNThreshold<>+0(SB)/8, $0x7c007c007c007c00
DATA ·f16NaNThreshold<>+8(SB)/8, $0x7c007c007c007c00
GLOBL ·f16NaNThreshold<>(SB), RODATA|NOPTR, $16

// func x86FeatureInfoECX() uint32
TEXT ·x86FeatureInfoECX(SB), NOSPLIT, $0-4
	MOVQ	BX, R11
	MOVL	$1, AX
	XORL	CX, CX
	CPUID
	MOVQ	R11, BX
	MOVL	CX, ret+0(FP)
	RET

// func f16LittleEndianToF32Asm(dst []float32, src []byte) int
// Converts len(dst) half values until the first 8-lane block containing a NaN.
// Caller guarantees len(dst)%8==0 and len(src)==2*len(dst). The return value
// is the number of elements converted, which equals len(dst) when the whole
// slice was processed in the vector loop.
TEXT ·f16LittleEndianToF32Asm(SB), NOSPLIT, $0-56
	MOVQ	dst_base+0(FP), SI
	MOVQ	dst_len+8(FP), CX
	MOVQ	src_base+24(FP), DI
	XORQ	AX, AX

	TESTQ	CX, CX
	JZ	f16_done

	VMOVDQU	·f16AbsMask<>+0(SB), X14
	VMOVDQU	·f16NaNThreshold<>+0(SB), X15

f16_loop8:
	VMOVDQU	(DI), X0
	VPAND	X14, X0, X1
	VPCMPGTW	X15, X1, X1
	VPMOVMSKB	X1, DX
	TESTL	DX, DX
	JNZ	f16_done

	VCVTPH2PS	X0, Y0
	VMOVUPS	Y0, (SI)
	ADDQ	$16, DI
	ADDQ	$32, SI
	ADDQ	$8, AX
	SUBQ	$8, CX
	JNZ	f16_loop8

f16_done:
	MOVQ	AX, ret+48(FP)
	VZEROUPPER
	RET
