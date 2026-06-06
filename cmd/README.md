# go-pherence `cmd/` — command index & kernel-migration map

Driver, benchmark, and prototype commands for the MilkV Jupiter 2 / SpaceMIT K3
(8× X100 + 8× A100 AI cores) inference work. This index exists so the kernels and
SIMD code currently living **inside** commands can be migrated into the right
go-pherence packages.

## Where kernels/SIMD should live

| Package | Owns |
|---|---|
| `backends/spacemit/ime2/` | IME2 `vmadot` matmul kernels (i8i4, i8i8, Q4K tiled, int8-group, argmax), repack, fused FFN/gate |
| `backends/simd/runtime/` | portable SIMD (sgemm/sdot/vec) for riscv64 / arm64 / amd64, Q8 quantise, byte copy |
| `npu/rvv/` | RVV matmul kernels (`ker_m4n32`, `dot`) |
| `backends/spacemit/tcm/` | TCM substrate + AI-core handshake (`/proc/set_ai_thread`, pair barriers, B-wave double-buffer) |
| `backends/k3/` | backend selection / placement / op dispatch |
| `runtime/` | decode loop, sampler, KV cache, speculative decoding |

## Migration priority (commands carrying kernel/SIMD source)

1. **`ime2run`** — the motherlode: 15 inline `.s` kernels + AI-core handshake +
   TCM B-wave + FFN/gate fuse + sampler. Source of truth for almost every kernel.
2. **`testi8i4`** — private duplicate of `k3_i8i4_go.s`; fold into the `ime2`
   package as a test once the canonical kernel moves there.
3. **`verifydot`** — dot-product correctness; becomes an `ime2`/`npu/rvv` test.
4. **`npu-tcm`** — canonical exerciser for the TCM/`npu` layer.

Everything else is a **thin driver/benchmark** over the backend packages (no
inline kernels) — keep them, just repoint at the consolidated packages.

## Commands documented (each has its own README)

**Kernel/SIMD source:** `ime2run`, `testi8i4`, `verifydot`, `ime2test`, `npu-tcm`

**K3 backend drivers & benchmarks:** `k3run`, `k3bench`, `k3llama`, `k3graphrun`,
`k3ggmlbench`, `k3ggmlplan`, `k3ffnblockbench`, `k3graphfusebench`, `k3qbench`,
`k3plandump`, `k3ortbench`, `k3ortlayerbench`

**Model runners / servers:** `qwen36run`, `llmserver`, `llmchat`, `llmgen`

**Speculative decoding:** `specbench`, `speccheck`

**MTP experiments (rejected on this CPU):** `qwenmtpmeta`, `qwenmtpsmoke`,
`qwenmtpsynth`, `gemma4mtpsmoke`

**GGUF / model inspection:** `ggufinspect`, `ggufsmoke`, `shapecheck`,
`embcheck`, `modelcoverage`, `tinydemo`

## Not documented here (separate audio / image / 3D / TTS workstreams)

`diarize-vtt`, `whisper`, `speakercheck`, `ideogram4gen`, `ideogram4inspect`,
`ideogram4vaesmoke`, `hy3dinspect`, `zimageinspect`, `qwen3ttsinspect`,
`lfm2inspect` — unrelated to the K3 kernel/SIMD effort.
