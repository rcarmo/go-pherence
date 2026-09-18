// Exact 16-row transpose packing. Full tiles read eight floats per row;
// four-element and scalar tails read only the remainder. No floating arithmetic.
#include "textflag.h"

#define PACK4(a, b, c, d, off) \
    MOVQ a, AX; \
    MOVQ b, BX; \
    MOVQ c, DX; \
    MOVQ d, SI; \
    VMOVUPS (AX)(R8*1), X0; \
    VMOVUPS (BX)(R8*1), X1; \
    VMOVUPS (DX)(R8*1), X2; \
    VMOVUPS (SI)(R8*1), X3; \
    VUNPCKLPS X1, X0, X4; \
    VUNPCKHPS X1, X0, X5; \
    VUNPCKLPS X3, X2, X6; \
    VUNPCKHPS X3, X2, X7; \
    VSHUFPS $0x44, X6, X4, X0; \
    VSHUFPS $0xEE, X6, X4, X1; \
    VSHUFPS $0x44, X7, X5, X2; \
    VSHUFPS $0xEE, X7, X5, X3; \
    VMOVUPS X0, off(DI); \
    VMOVUPS X1, off+64(DI); \
    VMOVUPS X2, off+128(DI); \
    VMOVUPS X3, off+192(DI)

#define PACK8(a, b, c, d, off) \
    MOVQ a, AX; \
    MOVQ b, BX; \
    MOVQ c, DX; \
    MOVQ d, SI; \
    VMOVUPS (AX)(R8*1), Y0; \
    VMOVUPS (BX)(R8*1), Y1; \
    VMOVUPS (DX)(R8*1), Y2; \
    VMOVUPS (SI)(R8*1), Y3; \
    VUNPCKLPS Y1, Y0, Y4; \
    VUNPCKHPS Y1, Y0, Y5; \
    VUNPCKLPS Y3, Y2, Y6; \
    VUNPCKHPS Y3, Y2, Y7; \
    VSHUFPS $0x44, Y6, Y4, Y0; \
    VSHUFPS $0xEE, Y6, Y4, Y1; \
    VSHUFPS $0x44, Y7, Y5, Y2; \
    VSHUFPS $0xEE, Y7, Y5, Y3; \
    VMOVUPS X0, off(DI); \
    VMOVUPS X1, off+64(DI); \
    VMOVUPS X2, off+128(DI); \
    VMOVUPS X3, off+192(DI); \
    VEXTRACTF128 $1, Y0, X4; \
    VEXTRACTF128 $1, Y1, X5; \
    VEXTRACTF128 $1, Y2, X6; \
    VEXTRACTF128 $1, Y3, X7; \
    VMOVUPS X4, off+256(DI); \
    VMOVUPS X5, off+320(DI); \
    VMOVUPS X6, off+384(DI); \
    VMOVUPS X7, off+448(DI)

#define PACK1(a, off) \
    MOVQ a, AX; \
    MOVL (AX)(R8*1), BX; \
    MOVL BX, off(DI)

TEXT ·packBNTAsm(SB), NOSPLIT, $0-144
    MOVQ k+128(FP), CX
    MOVQ bp+136(FP), DI
    XORQ R8, R8
    CMPQ CX, $8
    JL check4
loop8:
    PACK8(b0+0(FP), b1+8(FP), b2+16(FP), b3+24(FP), 0)
    PACK8(b4+32(FP), b5+40(FP), b6+48(FP), b7+56(FP), 16)
    PACK8(b8+64(FP), b9+72(FP), b10+80(FP), b11+88(FP), 32)
    PACK8(b12+96(FP), b13+104(FP), b14+112(FP), b15+120(FP), 48)
    ADDQ $32, R8
    ADDQ $512, DI
    SUBQ $8, CX
    CMPQ CX, $8
    JGE loop8
check4:
    CMPQ CX, $4
    JL tail
loop4:
    PACK4(b0+0(FP), b1+8(FP), b2+16(FP), b3+24(FP), 0)
    PACK4(b4+32(FP), b5+40(FP), b6+48(FP), b7+56(FP), 16)
    PACK4(b8+64(FP), b9+72(FP), b10+80(FP), b11+88(FP), 32)
    PACK4(b12+96(FP), b13+104(FP), b14+112(FP), b15+120(FP), 48)
    ADDQ $16, R8
    ADDQ $256, DI
    SUBQ $4, CX
    CMPQ CX, $4
    JGE loop4
tail:
    TESTQ CX, CX
    JLE done
loop1:
    PACK1(b0+0(FP), 0)
    PACK1(b1+8(FP), 4)
    PACK1(b2+16(FP), 8)
    PACK1(b3+24(FP), 12)
    PACK1(b4+32(FP), 16)
    PACK1(b5+40(FP), 20)
    PACK1(b6+48(FP), 24)
    PACK1(b7+56(FP), 28)
    PACK1(b8+64(FP), 32)
    PACK1(b9+72(FP), 36)
    PACK1(b10+80(FP), 40)
    PACK1(b11+88(FP), 44)
    PACK1(b12+96(FP), 48)
    PACK1(b13+104(FP), 52)
    PACK1(b14+112(FP), 56)
    PACK1(b15+120(FP), 60)
    ADDQ $4, R8
    ADDQ $64, DI
    DECQ CX
    JNZ loop1
done:
    VZEROUPPER
    RET
