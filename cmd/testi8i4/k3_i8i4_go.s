#include "textflag.h"

// func k3I8I4M1(a *byte, b *byte, c *float32, kBlks int, nBlks int)
TEXT ·k3I8I4M1(SB), NOSPLIT, $0-40
    MOV a+0(FP), X10
    MOV a+0(FP), X11
    ADD $6, X11
    MOV b+8(FP), X12
    MOV b+8(FP), X13
    ADD $128, X13
    MOV c+16(FP), X14
    MOV kBlks+24(FP), X15
    MOV nBlks+32(FP), X6
    WORD $0x008072d7
    WORD $0x5e00b057
    WORD $0x960230d7
    WORD $0x2e000157
    WORD $0x4a019057
    WORD $0x4a1190d7
loop:
    WORD $0x000072d7
    WORD $0x62868207
    ADD $640, X13
    WORD $0x007072d7
    WORD $0x02060f07
    ADD $64, X12
    WORD $0x006072d7
    WORD $0x02058187
    ADD $38, X11
    WORD $0x00052007
    WORD $0x00451383
    ADD $38, X10
    WORD $0x008072d7
    WORD $0x5e043e57
    WORD $0x007072d7
    WORD $0x02060e87
    ADD $576, X12
    WORD $0x000072d7
    WORD $0xa2323c57
    WORD $0x008072d7
    WORD $0xc3d06e57
    WORD $0x97c3ed57
    WORD $0x4231b42b
    WORD $0x438c352b
    WORD $0x4bc19857
    WORD $0x03003957
    WORD $0x03003a57
    WORD $0x03003b57
    WORD $0xd645082b
    WORD $0xd655092b
    WORD $0xd6650a2b
    WORD $0xd6750b2b
    WORD $0xcc44082b
    WORD $0xcc54092b
    WORD $0xcc640a2b
    WORD $0xcc740b2b
    WORD $0x67281c2b
    WORD $0x676a1d2b
    WORD $0x67ac282b
    WORD $0x00f072d7
    WORD $0xe3e81fd7
    WORD $0x010072d7
    WORD $0xb3f05157
    ADD $-1, X15
    BNE X15, X0, loop
    WORD $0x010372d7
    WORD $0x02076127
    RET
