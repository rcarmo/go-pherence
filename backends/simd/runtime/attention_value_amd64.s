#include "textflag.h"

TEXT ·attentionValueRowAsm(SB), NOSPLIT, $0-88
 MOVQ dst_base+0(FP),DI
 MOVQ probabilities_base+24(FP),SI
 MOVQ values_base+48(FP),R8
 MOVQ rows+72(FP),R9
 MOVQ width+80(FP),R10
 SHLQ $2,R10
 MOVQ R10,R11
columns:
 VXORPS Y0,Y0,Y0
 MOVQ SI,AX
 MOVQ R8,BX
 MOVQ R9,CX
sources:
 VBROADCASTSS (AX),Y1
 VMULPS (BX),Y1,Y1
 VADDPS Y1,Y0,Y0
 ADDQ $4,AX
 ADDQ R10,BX
 DECQ CX
 JNZ sources
 VMOVUPS Y0,(DI)
 ADDQ $32,DI
 ADDQ $32,R8
 SUBQ $32,R11
 JNZ columns
 VZEROUPPER
 RET
