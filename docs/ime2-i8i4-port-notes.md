# IME2 i8×i4 port notes

Source: llama.cpp PR #22863 (`spacemit-com/llama.cpp` commit `58745bdb86b759940577e0a8226cfbfa9987c286`).

## Data contract ported in Go

- Q4_K -> Q4_1x32 repack (`cmd/spacemit/ime2run/q4k_ai.go`):
  - `D[32]` fp16-rounded per output row and 32-column subblock
  - `ZP[32] = clamp(round(min/d), 0, 15)`
  - `QS[32][16]` packed q4 values in llama.cpp `make_block_q4_1x32` order
- q8 activation block (`cmd/spacemit/ime2run/q4k_llama_x32.go`):
  - 32-wide block
  - f32 scale
  - s16 negative sum
  - int8[32]

## Current status

- `IME2_Q4K_LLAMA_X32=1` routes through the llama.cpp-style data contract.
- It still unpacks to generic native `vmadot`, so it is not the final kernel.
- Token IDs with `IME2_Q4K_LLAMA_X32=1 IME2_LM_F32=1` currently diverge after the first tokens.

## Exported libggml-cpu kernel attempt

`libggml-cpu.so` exports:

```text
spacemit_kernels::ime2::gemm_kernel_i8i4(unsigned long, unsigned char const*, unsigned char const*, unsigned char const*, float*, unsigned long, unsigned long, unsigned long, unsigned long)
```

A cgo shim was added (`cmd/spacemit/ime2run/q4k_cshim.go`) and called under `IME2_Q4K_CSHIM=1`, but direct calls fault even with 128-byte aligned C buffers and AI-worker execution. Do not rely on this path except as a failed experiment.

## Instruction encodings obtained with GCC `-march=rv64gcv_xsmtvdotii`

```asm
smt.vmadotsu.hp v16,v10,v4,v1,0,i4  -> 0xd645082b
smt.vmadotu.hp  v16,v8,v4,v0,0,i4   -> 0xcc44082b
smt.vpack.vv    v24,v16,v18,1       -> 0x67281c2b
smt.vnpack4.vv  v8,v3,v3,3          -> 0x4231b42b
```

These can be used as `WORD` values in Go/Plan 9 asm for the direct port.

## Current direct Go/asm status

Added a direct Go/asm port of the PR's `gemm_kernel_i8i4_m1` active `#else` path:

- `cmd/spacemit/ime2run/k3_i8i4_go.s`
- env route: `IME2_Q4K_LLAMA_X32=1 IME2_Q4K_GOASM=1`
- synthetic all-ones kernel test passes in `cmd/spacemit/testi8i4`.

However, a randomized synthetic test and model-path `IME2_Q4K_COMPARE=1` still show mismatches vs the scalar `Q4_1x32` reference. The direct asm is executing on A100 and no longer faults, but the remaining issue is semantic/layout correspondence between:

- q8 activation byte layout consumed by `vnpack4.vv` + `vmadotsu.hp`/`vmadotu.hp`
- Q4_1x32 `qs` byte ordering consumed by `vl4r.v v4..v7`
- output lane order after `vpack.vv`

Do not port `m4` until `m1` matches the scalar reference on randomized `cmd/spacemit/testi8i4` cases.
