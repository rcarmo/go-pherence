//go:build amd64

#include "textflag.h"

// Test-only entry: temporarily change MXCSR, call affine, restore before Go.
// Kept in a separate symbol; linker excludes it from normal binaries.
TEXT ·affineTestControl(SB), NOSPLIT, $48-41
    STMXCSR 40(SP)
    MOVL 40(SP), AX
    ORL mask+32(FP), AX
    MOVL AX, 44(SP)
    MOVQ x_base+0(FP), AX
    MOVQ AX, 0(SP)
    MOVQ x_len+8(FP), AX
    MOVQ AX, 8(SP)
    MOVQ x_cap+16(FP), AX
    MOVQ AX, 16(SP)
    MOVL scale+24(FP), AX
    MOVL AX, 24(SP)
    MOVL shift+28(FP), AX
    MOVL AX, 28(SP)
    LDMXCSR 44(SP)
    CALL ·affineF32Asm(SB)
    LDMXCSR 40(SP)
    MOVBLZX 32(SP), AX
    MOVB AX, ret+40(FP)
    RET
