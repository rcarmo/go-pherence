#include "textflag.h"

// Equal non-zero lengths divisible by eight, validated by VecMulAddTo.
// Separate FMUL/FADD, never FMLA. All inputs loaded before aliased output stores.
TEXT ·vecMulAddAsm(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R3
    MOVD dst_len+8(FP), R2
    MOVD a_base+24(FP), R0
    MOVD b_base+48(FP), R1
loop:
    VLD1.P 32(R0), [V0.S4, V1.S4]
    VLD1.P 32(R1), [V2.S4, V3.S4]
    VLD1 (R3), [V4.S4, V5.S4]
    WORD $0x6e22dc00 // FMUL V0.4S, V0.4S, V2.4S
    WORD $0x6e23dc21 // FMUL V1.4S, V1.4S, V3.4S
    WORD $0x4e20d484 // FADD V4.4S, V4.4S, V0.4S
    WORD $0x4e21d4a5 // FADD V5.4S, V5.4S, V1.4S
    VST1.P [V4.S4, V5.S4], 32(R3)
    SUB $8, R2, R2
    CBNZ R2, loop
    RET
