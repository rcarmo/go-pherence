#include "textflag.h"

// func copyBytesRVV(dst, src *byte, n int)
// Copies n bytes in 128-byte RVV chunks. Caller guarantees n%128 == 0.
TEXT ·CopyBytesRVV(SB), NOSPLIT, $0-24
    MOV dst+0(FP), X10
    MOV src+8(FP), X11
    MOV n+16(FP), X12
    BEQ X12, X0, done_copy_rvv
    // VLEN=256: e8,m4 loads 4×32=128 bytes per iteration (correct for this hardware).
    // Old: e8,m1 loaded only 32 bytes/iter on VLEN=256 (4× too slow).
    WORD $0x0c2072d7        // vsetvli t0, zero, e8, m4, ta, ma (128B per iter on VLEN=256)
loop_copy_rvv:
    WORD $0x02058007        // vle8.v v0, (X11/a1)  [loads 128B into v0-v3 with m4]
    WORD $0x02050027        // vse8.v v0, (X10/a0)  [stores 128B]
    ADD $128, X11
    ADD $128, X10
    ADD $-128, X12
    BNE X12, X0, loop_copy_rvv
done_copy_rvv:
    RET
