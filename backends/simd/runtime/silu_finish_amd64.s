#include "textflag.h"
DATA ·siluFinishOne<>(SB)/4, $0x3f800000
GLOBL ·siluFinishOne<>(SB), RODATA|NOPTR, $4

// Validated equal lengths divisible by eight. Exact dst alias with gate/up
// is safe: all source vectors are loaded before the destination is written.
// Separate add/divide/multiply retains float32 rounding; no reciprocal estimate.
TEXT ·siluFinishAsm(SB), NOSPLIT, $0-96
 MOVQ dst_base+0(FP), DI
 MOVQ gate_base+24(FP), SI
 MOVQ up_base+48(FP), DX
 MOVQ exp_base+72(FP), BX
 MOVQ dst_len+8(FP), CX
 VBROADCASTSS ·siluFinishOne<>(SB), Y3
 TESTQ CX, CX
 JLE done
loop:
 VMOVUPS (SI), Y0
 VMOVUPS (DX), Y1
 VMOVUPS (BX), Y2
 VADDPS Y3, Y2, Y2
 VDIVPS Y2, Y0, Y0
 VMULPS Y1, Y0, Y0
 VMOVUPS Y0, (DI)
 ADDQ $32, DI
 ADDQ $32, SI
 ADDQ $32, DX
 ADDQ $32, BX
 SUBQ $8, CX
 JG loop
done:
 VZEROUPPER
 RET
