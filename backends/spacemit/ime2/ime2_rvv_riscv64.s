#include "textflag.h"

// func rvvMulVecVec(a *float32, b *float32, out *float32, n int)
TEXT ·rvvMulVecVec(SB), NOSPLIT, $0-32
    MOV  a+0(FP), X10
    MOV  b+8(FP), X11
    MOV  out+16(FP), X12
    MOV  n+24(FP), X13
    WORD $0x012072d7            // vsetvli t0, zero, e32, m4, tu, mu
loop:
    BEQ  X13, X0, done
    WORD $0x0126f2d7            // vsetvli t0, a3, e32, m4, tu, mu
    WORD $0x02056007            // vle32.v v0, (a0)
    WORD $0x0205e207            // vle32.v v4, (a1)
    WORD $0x92021057            // vfmul.vv v0, v0, v4
    WORD $0x02066027            // vse32.v v0, (a2)
    SLL  $2, X5, X6
    ADD  X6, X10, X10
    ADD  X6, X11, X11
    ADD  X6, X12, X12
    SUB  X5, X13, X13
    JMP  loop
done:
    RET

// func rvvBroadcastPack(src *byte, K int, dst *byte)
TEXT ·rvvBroadcastPack(SB), NOSPLIT, $0-24
    MOV  src+0(FP), X10
    MOV  K+8(FP), X11
    MOV  dst+16(FP), X12
    MOV  $8, X5
    WORD $0x0002f2d7            // vsetvli t0, t0, e8, m1
    SRLI $3, X11, X13
bploop:
    BEQ  X13, X0, bpdone
    WORD $0x02050007            // vle8.v v0, (a0)
    WORD $0x02060027            // vse8.v v0, (a2)
    ADD  $8, X12, X14
    WORD $0x02070027            // vse8.v v0, (a4)
    ADD  $8, X14, X14
    WORD $0x02070027            // vse8.v v0, (a4)
    ADD  $8, X14, X14
    WORD $0x02070027            // vse8.v v0, (a4)
    ADD  $8, X10, X10
    ADD  $32, X12, X12
    ADD  $-1, X13, X13
    JMP  bploop
bpdone:
    RET

// func rvvFindMaxAbs(x *float32, n int) float32
// Returns max(|x[i]|) using RVV.
TEXT ·rvvFindMaxAbs(SB), NOSPLIT, $0-20
    MOV  x+0(FP), X10         // a0 = x
    MOV  n+8(FP), X11         // a1 = n
    // Zero max accumulator
    WORD $0x012072d7            // vsetvli t0, zero, e32, m4
    WORD $0x5e004457            // vmv.v.i v8, 0  (max = 0)

findmax_loop:
    BEQ  X11, X0, findmax_done
    WORD $0x0125f2d7            // vsetvli t0, a1, e32, m4
    WORD $0x02056007            // vle32.v v0, (a0)
    WORD $0x2a001257            // vfabs.v v4, v0
    WORD $0x1a821457            // vfmax.vv v8, v8, v4
    SLL  $2, X5, X6            // bytes = vl*4
    ADD  X6, X10, X10
    SUB  X5, X11, X11
    JMP  findmax_loop

findmax_done:
    // Reduce v8 to scalar
    WORD $0x5e003057            // vmv.v.i v0, 0  (zero for reduce)  -- actually need vmv.s.x or use v0
    WORD $0x012072d7            // vsetvli t0, zero, e32, m4
    WORD $0x5e000057            // vmv.v.i v0, 0
    WORD $0x1e801057            // vfredmax.vs v0, v8, v0
    WORD $0x42001557            // vfmv.f.s fa0, v0
    // Store result
    MOVF F10, ret+16(FP)
    RET
