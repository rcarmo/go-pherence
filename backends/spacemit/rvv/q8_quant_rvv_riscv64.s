#include "textflag.h"

// func quantizeQ8Block32RVV(src *float32, dst *byte, divisor *float32)
// Q8 K32 block layout: [fp32 scale (4B)][i16 -sum (2B)][32×i8 (32B)] = 38B
//
// SpacemiT VLEN=256 path: e32,m4 for 32 elements.
// Encodings verified by disassembling gcc -march=rv64gcv output.
//
// VLEN=256 element counts:
//   e8,m1  → vl=32   e32,m1 → vl=8
//   e32,m4 → vl=32   e16,m2 → vl=32
//
// Trick: vsetvli a5,a5,e8,m1,ta,ma gets vl=32 into a5; subsequent
//   vsetvli zero,a5,e32,m4 use a5=32 as AVL to maintain vl=32 at m4.
//
// vle32.v with SEW=e8,m1 in effect: EEW=32, EMUL=EEW/SEW*LMUL=m4 → uses v4-v7.
// All subsequent ops: the vsetvli before each op sets the correct type.
TEXT ·QuantizeQ8Block32RVV(SB), NOSPLIT, $0-24
    MOV src+0(FP),      X10  // a0 = src
    MOV dst+8(FP),      X11  // a1 = dst
    MOV divisor+16(FP), X12  // a2 = &divisor

    // Load divisor float from pointer; set fa4 = 0.0f for zero-check
    WORD $0x00062507         // flw  fa0, 0(a2)   -- F10 = divisor
    WORD $0xf0000753         // fmv.w.x fa4, x0   -- F14 = 0.0f
    ADD  $6, X11, X13        // a3 = dst+6        (quant bytes offset)

    // --- vl = 32 via e8,m1 trick (VLMAX_e8m1 = VLEN/8 = 256/8 = 32) ---
    MOV  $32, X15            // a5 = 32 (requested AVL)
    WORD $0x0c07f7d7         // vsetvli a5, a5, e8, m1, ta, ma  → a5=vl=32

    // Load 32 fp32 values; EEW=32, EMUL=EEW/SEW*m1=m4 → occupies v4-v7
    WORD $0x02056207         // vle32.v v4, (a0)

    // --- Find max absolute value ---
    // Init m1 accumulator for reduction
    WORD $0x0d07f057         // vsetvli zero, a5, e32, m1, ta, ma
    WORD $0x5e0030d7         // vmv.v.i v1, 0

    // Switch to m4 for v4-v7 operations
    WORD $0x0d27f057         // vsetvli zero, a5, e32, m4, ta, ma
    WORD $0x2a421457         // vfabs.v   v8, v4       -- v8-v11 = |v4-v7|
    WORD $0x1e809457         // vfredmax.vs v8, v8, v1 -- v8[0] = max_abs
    WORD $0x428017d7         // vfmv.f.s  fa5, v8      -- F15 = max_abs

    // --- scale = max_abs / divisor; rep_scale = 1/scale or 0 ---
    WORD $0x18a7f553         // fdiv.s fa0, fa5, fa0   -- F10 = scale
    WORD $0xa0e52753         // feq.s  a4,  fa0, fa4   -- X14 = (scale==0.0f)
    BNE  X14, X0, skip_repscale

    // rep_scale = 1.0f / scale  (branch skipped if scale==0, leaving fa4=0.0f)
    MOV  $0x3f800000, X14    // 1.0f bit pattern
    WORD $0xf0070753         // fmv.w.x fa4, a4        -- F14 = 1.0f
    WORD $0x18a77753         // fdiv.s  fa4, fa4, fa0  -- F14 = 1.0f/scale

skip_repscale:
    // --- Quantize: multiply by rep_scale, store scale ---
    WORD $0x92475257         // vfmul.vf v4, v4, fa4   -- v4-v7 *= rep_scale
    WORD $0x00a5a027         // fsw fa0, 0(a1)          -- store fp32 scale at dst+0

    // --- Narrow f32(m4) → i16(m2) → i8(m1), in-place ---
    // Set destination SEW=e16,m2 (vl=32 maintained via a5)
    WORD $0x0c87f057         // vsetvli zero, a5, e16, m1, ta, ma  (init vmv below)
    WORD $0x5e0030d7         // vmv.v.i v1, 0           -- zero sum accumulator
    WORD $0x0c97f057         // vsetvli zero, a5, e16, m2, ta, ma
    WORD $0x4a489257         // vfncvt.x.f.w v4, v4     -- f32(m4)→i16(m2) in-place
    // Narrow i16(m2) → i8(m1) (dest vl maintained via rs1=zero → VLMAX_e8m1=32)
    WORD $0x0c007057         // vsetvli zero, zero, e8, m1, ta, ma
    WORD $0xb2403257         // vnsrl.wi v4, v4, 0      -- i16(m2)→i8(m1) in-place

    // --- Compute -sum and store ---
    WORD $0xc64080d7         // vwredsum.vs v1, v4, v1  -- v1[0] = sum(i8 → i16)
    WORD $0x0c907057         // vsetvli zero, zero, e16, m2, ta, ma
    WORD $0x42102757         // vmv.x.s a4, v1          -- X14 = sum
    WORD $0x40e0073b         // negw a4, a4             -- X14 = -sum
    WORD $0x00e59223         // sh a4, 4(a1)            -- store i16 at dst+4

    // --- Store 32 i8 bytes at dst+6 ---
    WORD $0x02068227         // vse8.v v4, (a3)
    RET
