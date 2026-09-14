#include "textflag.h"

// Internal equal-length, non-overlapping buffers. Keep separate multiplies
// and addition to preserve the three existing float32 rounding boundaries.
TEXT ·snakePostAsm(SB), NOSPLIT, $0-52
 MOVQ row_base+0(FP), AX
 MOVQ sine_base+24(FP), BX
 MOVQ row_len+8(FP), CX
 VBROADCASTSS scale+48(FP), Y2
loop:
 CMPQ CX, $8
 JL tail
 VMOVUPS (BX), Y0
 VMULPS Y0, Y0, Y0
 VMULPS Y2, Y0, Y0
 VADDPS (AX), Y0, Y0
 VMOVUPS Y0, (AX)
 ADDQ $32, AX
 ADDQ $32, BX
 SUBQ $8, CX
 JMP loop
tail:
 TESTQ CX, CX
 JZ done
 VMOVSS (BX), X0
 VMULSS X0, X0, X0
 VMULSS X2, X0, X0
 VADDSS (AX), X0, X0
 VMOVSS X0, (AX)
 ADDQ $4, AX
 ADDQ $4, BX
 DECQ CX
 JMP tail
done:
 VZEROUPPER
 RET
