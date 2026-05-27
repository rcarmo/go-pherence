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

// func rvvQuantizeF32ToI8(src *float32, scaleBits uint32, dst *byte, n int)
// Quantizes n float32s to int8: dst[i] = int8(src[i] * scale)
// scaleBits is the IEEE754 bits of the float32 scale factor.
TEXT ·rvvQuantizeF32ToI8(SB), NOSPLIT, $0-32
    MOV  src+0(FP), X10       // a0 = src
    MOVW scaleBits+8(FP), X5  // load scale bits
    WORD $0xf00282d3           // fmv.w.x ft5, t0 (move bits to float reg)
    MOV  dst+16(FP), X11      // a1 = dst
    MOV  n+24(FP), X12        // a2 = n

    // Process 8 floats at a time (e32,m1 → 8 with VLEN=256)
quant_loop:
    BEQ  X12, X0, quant_done
    MOV  $8, X5
    WORD $0x0d02f2d7            // vsetvli t0, t0, e32, m1, ta, ma
    WORD $0x02056007            // vle32.v v0, (a0)
    WORD $0x9202d057            // vfmul.vf v0, v0, ft5
    WORD $0x4a009057            // vfcvt.x.f.v v0, v0 (float→int32, rounding)
    // Narrow int32→int16
    WORD $0x0cf2f2d7            // vsetvli t0, t0, e16, mf2, ta, ma
    WORD $0xb2003257            // vnsrl.wi v4, v0, 0, v0, 0
    // Narrow int16→int8
    WORD $0x0c62f2d7            // vsetvli t0, t0, e8, mf4, ta, ma
    WORD $0xb2403457            // vnsrl.wi v8, v4, 0, v4, 0
    // Store 8 int8s
    MOV  $8, X5
    WORD $0x0002f2d7            // vsetvli t0, t0, e8, m1 (vl=8)
    WORD $0x02058427            // vse8.v v8, (a1)
    // Advance
    ADD  $32, X10, X10         // src += 8 floats = 32 bytes
    ADD  $8, X11, X11          // dst += 8 bytes
    ADD  $-8, X12, X12         // n -= 8
    JMP  quant_loop

quant_done:
    RET
// func rvvDotF32(a *float32, b *float32, n int) float32
// Returns sum(a[i]*b[i]) for i in 0..n-1 using RVV e32,m4.
// Rolling accumulator: each vfmacc.vv accumulates the next vl-element
// chunk into the same v16 slots, so after all chunks v16[j] holds
// sum of a[j + k*vlmax]*b[j + k*vlmax] over all k.  The final
// vfredusum then collapses those to a scalar.
// FP layout: a+0(FP)[8] b+8(FP)[8] n+16(FP)[8] ret+24(FP)[4]
TEXT ·rvvDotF32(SB), NOSPLIT, $0-28
    MOV  a+0(FP),  X10        // a0 = ptr to a
    MOV  b+8(FP),  X11        // a1 = ptr to b
    MOV  n+16(FP), X12        // a2 = n
    // Initialise accumulator v16 (m4 group = v16..v19) to 0.0
    WORD $0x012072d7            // vsetvli t0, zero, e32, m4, tu, mu
    WORD $0x5e003857            // vmv.v.i v16, 0
dot_loop:
    BEQ  X12, X0, dot_done
    WORD $0x012672d7            // vsetvli t0, a2, e32, m4, tu, mu
    WORD $0x02056007            // vle32.v v0, (a0)
    WORD $0x0205e407            // vle32.v v8, (a1)
    WORD $0xb2801857            // vfmacc.vv v16, v0, v8
    SLL  $2, X5, X6             // X6 = vl * 4 bytes
    ADD  X6, X10, X10
    ADD  X6, X11, X11
    SUB  X5, X12, X12
    JMP  dot_loop
dot_done:
    // Reset vl to max so the reduction covers all vlmax slots of v16
    WORD $0x012072d7            // vsetvli t0, zero, e32, m4, tu, mu
    WORD $0x5e003457            // vmv.v.i v8, 0   (reduction identity = 0.0)
    WORD $0x07041057            // vfredusum.vs v0, v16, v8
    WORD $0x42001557            // vfmv.f.s fa0, v0
    MOVF F10, ret+24(FP)
    RET

// func rvvScaleAccF32(acc *float32, src *float32, scale float32, n int)
// acc[i] += scale * src[i] for i in 0..n-1 using RVV e32,m4.
// FP layout: acc+0(FP)[8] src+8(FP)[8] scale+16(FP)[4] _pad[4] n+24(FP)[8]
TEXT ·rvvScaleAccF32(SB), NOSPLIT, $0-32
    MOV  acc+0(FP),   X10      // a0 = acc
    MOV  src+8(FP),   X11      // a1 = src
    MOVW scale+16(FP), X5      // t0 = scale bits (32-bit IEEE754)
    WORD $0xf00282d3            // fmv.w.x ft5, t0  (int bits → float reg)
    MOV  n+24(FP),    X12      // a2 = n
scaleacc_loop:
    BEQ  X12, X0, scaleacc_done
    WORD $0x0d2672d7            // vsetvli t0, a2, e32, m4, ta, ma
    WORD $0x02056007            // vle32.v v0, (a0)   load acc chunk
    WORD $0x0205e407            // vle32.v v8, (a1)   load src chunk
    WORD $0xb282d057            // vfmacc.vf v0, ft5, v8   v0 += scale * v8
    WORD $0x02056027            // vse32.v v0, (a0)   write back
    SLL  $2, X5, X6             // X6 = vl * 4 bytes
    ADD  X6, X10, X10
    ADD  X6, X11, X11
    SUB  X5, X12, X12
    JMP  scaleacc_loop
scaleacc_done:
    RET
