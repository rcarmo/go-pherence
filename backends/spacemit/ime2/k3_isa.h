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
