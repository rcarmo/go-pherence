# go-pherence `cmd/` — command index

Driver, benchmark, and prototype commands for the MilkV Jupiter 2 / SpaceMIT K3
(8× X60 RISC-V, RVV 1.0 + IME int8) inference work.

Commands are grouped by workstream. None of them are imported by other packages,
so the grouping is purely organisational.

## Groups

| Dir | Workstream | Commands |
|---|---|---|
| `k3/` | SpaceMIT K3 backend, kernels & benchmarks | `ime2run`, `ime2test`, `npu-tcm`, `testi8i4`, `verifydot`, `k3run`, `k3bench`, `k3llama`, `k3graphrun`, `k3ggmlbench`, `k3ggmlplan`, `k3ffnblockbench`, `k3graphfusebench`, `k3qbench`, `k3plandump`, `k3ortbench`, `k3ortlayerbench` |
| `llm/` | LLM serving / generation + speculative decoding | `llmserver`, `llmchat`, `llmgen`, `specbench`, `speccheck` |
| `qwen/` | Qwen model runners & MTP experiments | `qwen36run`, `qwen3ttsinspect`, `qwenmtpmeta`, `qwenmtpsmoke`, `qwenmtpsynth` |
| `audio/` | Speech transcription / diarization | `whisper`, `diarize-vtt`, `speakercheck` |
| `image/` | Image / 3D generation & inspection | `ideogram4gen`, `ideogram4inspect`, `ideogram4vaesmoke`, `zimageinspect`, `hy3dinspect` |
| `models/` | GGUF / model format inspection & coverage | `ggufinspect`, `ggufsmoke`, `lfm2inspect`, `modelcoverage`, `embcheck`, `shapecheck`, `gemma4mtpsmoke` |
| _(top level)_ | misc | `tinydemo` |

## Kernel migration — done

The kernels and SIMD that used to live **inside** `cmd/k3/ime2run` (a 69-file
`package main`) have been filed into their proper backend packages:

| Package | Owns |
|---|---|
| `backends/spacemit/ime2/` | IME2 `vmadot` int8 GEMM kernels (i8i4, i8i8, Q4K tiled, int8-group, argmax) + generic `WorkerPool` |
| `backends/spacemit/rvv/` | RVV 1.0 SIMD kernels (int8 GEMM, W4A8, byte-copy, q8 quant) |
| `backends/spacemit/tcm/` | TCM (on-chip SRAM) driver |
| `backends/spacemit/aicpu/` | The pure-Go transformer inference engine (decode loop + q4k/q6k/i8i4 kernels); `aipool/` worker pool, `config/` flags |
| `backends/spacemit/board/` | High-level ORT/Vulkan backend selection / dispatch |

See `backends/spacemit/README.md` for the package layout and dependency map.
The `cmd/k3/ime2run` command is now a thin wrapper around
`backends/spacemit/aicpu`.
