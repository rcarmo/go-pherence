// k3_isa.h — reusable macros for the SpaceMIT K3 / A100 IME2 extended
// vector instructions (the "AI-CPU" custom-opcode ISA used by go-pherence's
// RISC-V kernels).
//
// These instructions are NOT understood by the Go riscv64 assembler, so the
// kernels hand-encode them as `WORD $0x...`. This header replaces the scattered
// raw hex with named, self-documenting macros that compute the encoding from
// the operand register numbers, so future kernels (and the migration of the
// cmd/ime2run prototypes into this package) can read and write them safely.
//
// Usage:
//     #include "textflag.h"
//     #include "k3_isa.h"
//     ...
//     VMADOT_SS(28, 0, 1)        // v28 += v0(signed) . v1(signed)
//     VMADOT_SU(28, 3, 4)        // v28 += v3(signed act) . v4(unsigned 4-bit wt)
//     VPACK(16, 0, 2, 3)         // pack mode 3
//
// Register arguments are plain integers 0..31 (e.g. 28 means v28).
//
// ---------------------------------------------------------------------------
// Encoding (RVV-style, Custom-1 major opcode 0x2b):
//
//   31      26 25 24   20 19   15 14 12 11    7 6      0
//   +---------+--+-------+-------+------+-------+--------+
//   | funct6  |vm|  vs2  |  vs1  |funct3|  vd   |  0x2b  |
//   +---------+--+-------+-------+------+-------+--------+
//
//   word = base | (vd<<7) | (funct3<<12) | (vs1<<15) | (vs2<<20)
//
// where `base` carries funct6, vm and (for fixed-mode ops) funct3.
//
// funct3 semantics for the integer dot product `vmadot` = operand signedness:
//   0 = UU (unsigned x unsigned)
//   1 = SU (signed   x unsigned)   <- signed activation x unsigned 4-bit weight
//   2 = US (unsigned x signed)
//   3 = SS (signed   x signed)     <- the common int8 x int8 path
//
// Verified: these macros reproduce all 57 hand-encoded extended instructions
// currently in backends/spacemit/ime2 and cmd/ime2run, byte-for-byte.
// ---------------------------------------------------------------------------

// vmadot vd, vs1, vs2 — integer matrix dot-accumulate (funct6=0x38, vm=1).
// funct3 selects operand signedness; pick the named variant you need.
#define VMADOT_UU(vd, vs1, vs2) WORD $(0xe200002b | ((vd)<<7) | ((vs1)<<15) | ((vs2)<<20))
#define VMADOT_SU(vd, vs1, vs2) WORD $(0xe200102b | ((vd)<<7) | ((vs1)<<15) | ((vs2)<<20))
#define VMADOT_US(vd, vs1, vs2) WORD $(0xe200202b | ((vd)<<7) | ((vs1)<<15) | ((vs2)<<20))
#define VMADOT_SS(vd, vs1, vs2) WORD $(0xe200302b | ((vd)<<7) | ((vs1)<<15) | ((vs2)<<20))
// Default vmadot == signed x signed (most kernels).
#define VMADOT(vd, vs1, vs2)    VMADOT_SS(vd, vs1, vs2)

// vmadotu.hp  vd, vs1, vs2 — unsigned x unsigned, fp16 accumulate (funct6=0x33).
#define VMADOTU_HP(vd, vs1, vs2)  WORD $(0xcc00002b | ((vd)<<7) | ((vs1)<<15) | ((vs2)<<20))
// vmadotsu.hp vd, vs1, vs2 — signed x unsigned, fp16 accumulate (funct6=0x35).
#define VMADOTSU_HP(vd, vs1, vs2) WORD $(0xd600002b | ((vd)<<7) | ((vs1)<<15) | ((vs2)<<20))

// vnpack4.vv vd, vs1, vs2 — 4:1 narrowing pack (funct6=0x10, funct3=3).
#define VNPACK4(vd, vs1, vs2)     WORD $(0x4200302b | ((vd)<<7) | ((vs1)<<15) | ((vs2)<<20))

// vupack.vv vd, vs1, vs2 — widening unpack (funct6=0x19, funct3=5).
#define VUPACK(vd, vs1, vs2)      WORD $(0x6600502b | ((vd)<<7) | ((vs1)<<15) | ((vs2)<<20))

// vpack.vv vd, vs1, vs2, mode — interleave/pack, `mode` (1..3) in funct3
// (funct6=0x19, vm=1).
#define VPACK(vd, vs1, vs2, mode) WORD $(0x6600002b | ((mode)<<12) | ((vd)<<7) | ((vs1)<<15) | ((vs2)<<20))

// ---------------------------------------------------------------------------
// Standard RVV/Zvfh helper macros used by K3-native FP16 kernels.
// These are standard RVV instructions (not Custom-1 IME2 opcodes), but Go's
// riscv64 assembler still cannot spell them, so keep the encodings centralized
// here alongside the custom-op macros. Encodings are GNU as/objdump verified
// with -march=rv64gcv_zvfh.
// ---------------------------------------------------------------------------

// Fixed-vl setup used by FP16 tile kernels (t6/X31 holds tile N).
#define K3_VSETVLI_E32_M2_16        WORD $0x0d1ff057 // vsetvli zero, t6, e32, m2, ta, ma
#define K3_VSETVLI_E16_M1_16        WORD $0x0c8ff057 // vsetvli zero, t6, e16, m1, ta, ma
#define K3_VSETVLI_E32_M4_32        WORD $0x0d2ff057 // vsetvli zero, t6, e32, m4, ta, ma
#define K3_VSETVLI_E16_M2_32        WORD $0x0c9ff057 // vsetvli zero, t6, e16, m2, ta, ma

// Variable-vl setup used by FP16 dot/convert kernels (a2/X12 holds remaining length).
#define K3_VSETVLI_E32_M4_ZERO_TU_MU WORD $0x012072d7 // vsetvli t0, zero, e32, m4, tu, mu
#define K3_VSETVLI_E16_M2_A2_TU_MU   WORD $0x009672d7 // vsetvli t0, a2,   e16, m2, tu, mu
#define K3_VSETVLI_E32_M4_A2_TA_MA   WORD $0x0d2672d7 // vsetvli t0, a2,   e32, m4, ta, ma
#define K3_VSETVLI_E16_M2_A2_TA_MA   WORD $0x0c9672d7 // vsetvli t0, a2,   e16, m2, ta, ma

// Vector zeroing / identity values.
#define K3_VMV_V_I_V8_0             WORD $0x5e003457 // vmv.v.i v8,  0
#define K3_VMV_V_I_V10_0            WORD $0x5e003557 // vmv.v.i v10, 0
#define K3_VMV_V_I_V12_0            WORD $0x5e003657 // vmv.v.i v12, 0
#define K3_VMV_V_I_V14_0            WORD $0x5e003757 // vmv.v.i v14, 0
#define K3_VMV_V_I_V16_0            WORD $0x5e003857 // vmv.v.i v16, 0
#define K3_VMV_V_I_V20_0            WORD $0x5e003a57 // vmv.v.i v20, 0

// FP16 vector loads/broadcasts.
#define K3_VLE16_V0_A1              WORD $0x0205d007 // vle16.v  v0, (a1)
#define K3_VLE16_V0_A0              WORD $0x02055007 // vle16.v  v0, (a0)
#define K3_VLE16_V8_A1              WORD $0x0205d407 // vle16.v  v8, (a1)
#define K3_VLE32_V8_A0              WORD $0x02056407 // vle32.v  v8, (a0)
#define K3_VSE16_V0_A1              WORD $0x0205d027 // vse16.v  v0, (a1)
// Initial correctness kernels used vlse16 with zero stride for FP16 scalar
// broadcast. Prefer the cheaper lhu + vmv.v.x sequence below when possible.
#define K3_VLSE16_V2_A0_ZERO        WORD $0x0a055107 // vlse16.v v2,  (a0), zero
#define K3_VLSE16_V4_A0_ZERO        WORD $0x0a055207 // vlse16.v v4,  (a0), zero
#define K3_VLSE16_V4_T0_ZERO        WORD $0x0a02d207 // vlse16.v v4,  (t0), zero
#define K3_VLSE16_V4_T1_ZERO        WORD $0x0a035207 // vlse16.v v4,  (t1), zero
#define K3_VLSE16_V4_T2_ZERO        WORD $0x0a03d207 // vlse16.v v4,  (t2), zero
#define K3_VLSE16_V6_T1_ZERO        WORD $0x0a035307 // vlse16.v v6,  (t1), zero
#define K3_VLSE16_V16_T2_ZERO       WORD $0x0a03d807 // vlse16.v v16, (t2), zero
#define K3_LHU_T3_A0                WORD $0x00055e03 // lhu t3, 0(a0)
#define K3_LHU_T4_T0                WORD $0x0002de83 // lhu t4, 0(t0)
#define K3_LHU_T5_T1                WORD $0x00035f03 // lhu t5, 0(t1)
#define K3_LHU_T3_T2                WORD $0x0003de03 // lhu t3, 0(t2)
#define K3_VMV_V_X_V4_T3            WORD $0x5e0e4257 // vmv.v.x v4,  t3
#define K3_VMV_V_X_V4_T4            WORD $0x5e0ec257 // vmv.v.x v4,  t4
#define K3_VMV_V_X_V4_T5            WORD $0x5e0f4257 // vmv.v.x v4,  t5
#define K3_VMV_V_X_V6_T4            WORD $0x5e0ec357 // vmv.v.x v6,  t4
#define K3_VMV_V_X_V16_T5           WORD $0x5e0f4857 // vmv.v.x v16, t5
#define K3_VMV_V_X_V20_T3           WORD $0x5e0e4a57 // vmv.v.x v20, t3

// FP16 widening FMA, f16*f16 -> f32 accumulator.
#define K3_VFWMACC_VV_V16_V0_V8     WORD $0xf2801857 // vfwmacc.vv v16, v0,  v8
#define K3_VFWMACC_VV_V8_V2_V0      WORD $0xf2011457 // vfwmacc.vv v8,  v2,  v0
#define K3_VFWMACC_VV_V10_V4_V0     WORD $0xf2021557 // vfwmacc.vv v10, v4,  v0
#define K3_VFWMACC_VV_V12_V6_V0     WORD $0xf2031657 // vfwmacc.vv v12, v6,  v0
#define K3_VFWMACC_VV_V14_V16_V0    WORD $0xf2081757 // vfwmacc.vv v14, v16, v0
#define K3_VFWMACC_VV_V8_V4_V0      WORD $0xf2021457 // vfwmacc.vv v8,  v4,  v0
#define K3_VFWMACC_VV_V14_V4_V0     WORD $0xf2021757 // vfwmacc.vv v14, v4,  v0
#define K3_VFWMACC_VV_V12_V4_V0     WORD $0xf2021657 // vfwmacc.vv v12, v4,  v0
#define K3_VFWMACC_VV_V16_V4_V0     WORD $0xf2021857 // vfwmacc.vv v16, v4,  v0
#define K3_VFWMACC_VV_V20_V4_V0     WORD $0xf2021a57 // vfwmacc.vv v20, v4,  v0

// FP32/FP16 conversion and FP32 reduction/stores for FP16 kernels.
#define K3_VFNCVT_F_F_W_V0_V8       WORD $0x4a8a1057 // vfncvt.f.f.w v0, v8 (f32 -> f16)
#define K3_VFREDUSUM_VS_V0_V16_V8   WORD $0x07041057 // vfredusum.vs v0, v16, v8
#define K3_VFMV_F_S_FA0_V0          WORD $0x42001557 // vfmv.f.s fa0, v0
#define K3_VSE32_V8_A2              WORD $0x02066427 // vse32.v v8,  (a2)
#define K3_VSE32_V10_A2             WORD $0x02066527 // vse32.v v10, (a2)
#define K3_VSE32_V12_A2             WORD $0x02066627 // vse32.v v12, (a2)
#define K3_VSE32_V14_A2             WORD $0x02066727 // vse32.v v14, (a2)
#define K3_VSE32_V16_A2             WORD $0x02066827 // vse32.v v16, (a2)
#define K3_VSE32_V20_A2             WORD $0x02066a27 // vse32.v v20, (a2)
