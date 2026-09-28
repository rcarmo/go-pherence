#include "textflag.h"

// Equal non-zero lengths divisible by eight, validated by VecMulAddTo.
// Separate VMULPS/VADDPS retain both F32 rounding steps. Exact aliasing is safe
// because all three inputs for a block are loaded before its store.
TEXT ·vecMulAddAsm(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DI
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DX
loop:
    VMOVUPS (SI), Y0
    VMOVUPS (DX), Y1
    VMOVUPS (DI), Y2
    VMULPS Y1, Y0, Y0
    VADDPS Y0, Y2, Y2
    VMOVUPS Y2, (DI)
    ADDQ $32, SI
    ADDQ $32, DX
    ADDQ $32, DI
    SUBQ $8, CX
    JNZ loop
    VZEROUPPER
    RET
