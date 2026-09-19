## Repository safety audit: package coverage

Companion to [the audit report](repository-safety-audit-20260919.md).

This table records all 161 host-listed packages, not a safety clearance.
"Boundary inspection" means selected functions/sections in the listed files were
read for ownership, bounds or lifecycle behaviour; it does **not** mean the whole
file or package was reviewed. Blank source coverage is explicit outstanding work.
The race column is from the NVIDIA-disabled host sweep; a passing package may
contain skipped GPU, K3 or missing-fixture tests. Foreign-only files were not
executed. Test-source files explain regressions rather than broaden source coverage.

| Package | Host race result | Source boundary inspection (selected sections) |
|---|---|---|
| `backends/cuda/ptx` | pass | Inventory/tests only; source review outstanding |
| `backends/ggmlcompute` | no test files | Inventory/tests only; source review outstanding |
| `backends/ggmlexec` | no test files | Inventory/tests only; source review outstanding |
| `backends/ggmlgraph` | no test files | Inventory/tests only; source review outstanding |
| `backends/ggmlquant` | no test files | Inventory/tests only; source review outstanding |
| `backends/internal/ggmlutil` | no test files | Inventory/tests only; source review outstanding |
| `backends/llamagraph` | no test files | Inventory/tests only; source review outstanding |
| `backends/mlx` | pass | Inventory/tests only; source review outstanding |
| `backends/nvidia` | no test files | Inventory/tests only; source review outstanding |
| `backends/nvidia/internal/debuglog` | no test files | Inventory/tests only; source review outstanding |
| `backends/nvidia/ioctl` | pass | Inventory/tests only; source review outstanding |
| `backends/nvidia/ptx` | no test files | `attention.go`, `conversion.go`, `norm.go`, `sgemm_compensated.go` |
| `backends/nvidia/ptx/bf16` | no test files | Inventory/tests only; source review outstanding |
| `backends/nvidia/ptx/fp8` | no test files | Inventory/tests only; source review outstanding |
| `backends/nvidia/ptx/ideogram` | no test files | Inventory/tests only; source review outstanding |
| `backends/nvidia/ptx/mlx` | no test files | Inventory/tests only; source review outstanding |
| `backends/nvidia/ptx/nvfp4` | no test files | Inventory/tests only; source review outstanding |
| `backends/nvidia/ptx/q4` | no test files | Inventory/tests only; source review outstanding |
| `backends/nvidia/ptx/q5` | no test files | Inventory/tests only; source review outstanding |
| `backends/nvidia/ptx/q8` | no test files | Inventory/tests only; source review outstanding |
| `backends/nvidia/runtime` | pass | `driver_scope.go`, `driver_call_inventory_test.go`, `driver_scope_test.go`, `context_boundary_test.go`, `runtime.go`, `streams.go`, `compiler.go`, `compiler_test.go`, `devbuf.go`, `argmax.go`, `bf16_native.go`, `mega_module.go`, `module_state.go`, `attention_splitkv.go`, `bf16_projection.go` |
| `backends/placement` | pass | Inventory/tests only; source review outstanding |
| `backends/simd` | no test files | Inventory/tests only; source review outstanding |
| `backends/simd/fft` | pass | Inventory/tests only; source review outstanding |
| `backends/simd/kernels` | pass | Inventory/tests only; source review outstanding |
| `backends/simd/quant/bf16` | pass | Inventory/tests only; source review outstanding |
| `backends/simd/quant/fp8` | pass | Inventory/tests only; source review outstanding |
| `backends/simd/quant/nvfp4` | pass | Inventory/tests only; source review outstanding |
| `backends/simd/quant/q4` | pass | Inventory/tests only; source review outstanding |
| `backends/simd/runtime` | pass | `gemm_parallel.go` |
| `backends/spacemit/aicpu` | no test files | Inventory/tests only; source review outstanding |
| `backends/spacemit/aicpu/aipool` | no test files | `ai_pool.go`, `ai_pool_new.go`, `ai_thread.go` |
| `backends/spacemit/aicpu/config` | no test files | Inventory/tests only; source review outstanding |
| `backends/spacemit/aicpu/q4kcshim` | no test files | Inventory/tests only; source review outstanding |
| `backends/spacemit/board` | no test files | Inventory/tests only; source review outstanding |
| `backends/spacemit/ime2` | pass | `worker_pool.go` |
| `backends/spacemit/inference` | pass | Inventory/tests only; source review outstanding |
| `backends/spacemit/ort` | pass | Inventory/tests only; source review outstanding |
| `backends/spacemit/rvv` | pass | Inventory/tests only; source review outstanding |
| `backends/spacemit/tcm` | pass | Inventory/tests only; source review outstanding |
| `backends/vulkan` | pass | `vulkan_init.go`, `vulkan_ops.go`, `vulkan_wrapper_test.go`, `vulkan_buf.go` |
| `cmd/audio/diarize-vtt` | pass | Inventory/tests only; source review outstanding |
| `cmd/audio/internal/whisperflags` | pass | Inventory/tests only; source review outstanding |
| `cmd/audio/moss-transcribe` | pass | Inventory/tests only; source review outstanding |
| `cmd/audio/omnivoice` | pass | Inventory/tests only; source review outstanding |
| `cmd/audio/speakercheck` | no test files | Inventory/tests only; source review outstanding |
| `cmd/audio/speechjob` | pass | Inventory/tests only; source review outstanding |
| `cmd/audio/speechjobserve` | pass | Inventory/tests only; source review outstanding |
| `cmd/audio/whisper` | pass | Inventory/tests only; source review outstanding |
| `cmd/audio/whisperffndiag` | no test files | Inventory/tests only; source review outstanding |
| `cmd/diffusiongemmainspect` | no test files | Inventory/tests only; source review outstanding |
| `cmd/diffusiongemmarun` | pass | Inventory/tests only; source review outstanding |
| `cmd/diffusiongemmaserve` | no test files | Inventory/tests only; source review outstanding |
| `cmd/diffusiongemmaserver` | no test files | `main.go` |
| `cmd/gliner2` | pass | Inventory/tests only; source review outstanding |
| `cmd/image/hy3dinspect` | no test files | Inventory/tests only; source review outstanding |
| `cmd/image/ideogram4gen` | no test files | Inventory/tests only; source review outstanding |
| `cmd/image/ideogram4inspect` | no test files | Inventory/tests only; source review outstanding |
| `cmd/image/ideogram4vaeprobe` | no test files | Inventory/tests only; source review outstanding |
| `cmd/image/ideogram4vaesmoke` | no test files | Inventory/tests only; source review outstanding |
| `cmd/image/internal/k3flags` | no test files | Inventory/tests only; source review outstanding |
| `cmd/image/zimageinspect` | no test files | Inventory/tests only; source review outstanding |
| `cmd/internal/dgflags` | no test files | Inventory/tests only; source review outstanding |
| `cmd/internal/testexec` | no test files | Inventory/tests only; source review outstanding |
| `cmd/jevlike` | pass | `frozen.go` |
| `cmd/llm/internal/pathutil` | no test files | Inventory/tests only; source review outstanding |
| `cmd/llm/internal/promptfile` | no test files | Inventory/tests only; source review outstanding |
| `cmd/llm/llmchat` | no test files | Inventory/tests only; source review outstanding |
| `cmd/llm/llmgen` | pass | Inventory/tests only; source review outstanding |
| `cmd/llm/llmserver` | pass | `main.go` |
| `cmd/llm/servebench` | pass | Inventory/tests only; source review outstanding |
| `cmd/llm/specbench` | pass | Inventory/tests only; source review outstanding |
| `cmd/llm/speccheck` | pass | Inventory/tests only; source review outstanding |
| `cmd/minicpmvinspect` | no test files | Inventory/tests only; source review outstanding |
| `cmd/models/embcheck` | no test files | Inventory/tests only; source review outstanding |
| `cmd/models/gemma4mtpparity` | pass | Inventory/tests only; source review outstanding |
| `cmd/models/gemma4mtpsmoke` | pass | Inventory/tests only; source review outstanding |
| `cmd/models/ggufinspect` | pass | Inventory/tests only; source review outstanding |
| `cmd/models/ggufsmoke` | pass | Inventory/tests only; source review outstanding |
| `cmd/models/lfm2inspect` | pass | Inventory/tests only; source review outstanding |
| `cmd/models/modelcoverage` | pass | Inventory/tests only; source review outstanding |
| `cmd/models/shapecheck` | no test files | Inventory/tests only; source review outstanding |
| `cmd/qwen/qwen36run` | pass | Inventory/tests only; source review outstanding |
| `cmd/qwen/qwen3ttsinspect` | pass | Inventory/tests only; source review outstanding |
| `cmd/qwen/qwenmtpmeta` | pass | Inventory/tests only; source review outstanding |
| `cmd/qwen/qwenmtpsmoke` | pass | Inventory/tests only; source review outstanding |
| `cmd/qwen/qwenmtpsynth` | pass | Inventory/tests only; source review outstanding |
| `cmd/spacemit/ime2run` | no test files | Inventory/tests only; source review outstanding |
| `cmd/spacemit/ime2test` | no test files | Inventory/tests only; source review outstanding |
| `cmd/spacemit/npu-tcm` | no test files | Inventory/tests only; source review outstanding |
| `cmd/spacemit/spacemit_bench` | no test files | Inventory/tests only; source review outstanding |
| `cmd/spacemit/spacemit_ffnblockbench` | no test files | Inventory/tests only; source review outstanding |
| `cmd/spacemit/spacemit_ggmlbench` | no test files | Inventory/tests only; source review outstanding |
| `cmd/spacemit/spacemit_ggmlplan` | no test files | Inventory/tests only; source review outstanding |
| `cmd/spacemit/spacemit_graphfusebench` | no test files | Inventory/tests only; source review outstanding |
| `cmd/spacemit/spacemit_graphrun` | no test files | Inventory/tests only; source review outstanding |
| `cmd/spacemit/spacemit_llama` | no test files | Inventory/tests only; source review outstanding |
| `cmd/spacemit/spacemit_ortbench` | no test files | Inventory/tests only; source review outstanding |
| `cmd/spacemit/spacemit_ortlayerbench` | no test files | Inventory/tests only; source review outstanding |
| `cmd/spacemit/spacemit_plandump` | no test files | Inventory/tests only; source review outstanding |
| `cmd/spacemit/spacemit_qbench` | no test files | Inventory/tests only; source review outstanding |
| `cmd/spacemit/spacemit_run` | no test files | Inventory/tests only; source review outstanding |
| `cmd/spacemit/testi8i4` | no test files | Inventory/tests only; source review outstanding |
| `cmd/spacemit/verifydot` | no test files | Inventory/tests only; source review outstanding |
| `cmd/tinydemo` | no test files | Inventory/tests only; source review outstanding |
| `docs` | pass | Inventory/tests only; source review outstanding |
| `gpu` | pass | Inventory/tests only; source review outstanding |
| `half` | pass | Inventory/tests only; source review outstanding |
| `internal/checked` | no test files | Inventory/tests only; source review outstanding |
| `internal/floatcmp` | no test files | Inventory/tests only; source review outstanding |
| `internal/ggmlfp16` | pass | Inventory/tests only; source review outstanding |
| `internal/modelcoverage` | no test files | Inventory/tests only; source review outstanding |
| `loader/audio` | pass | `wav.go`, `wav_invalid_test.go` |
| `loader/audio/media` | pass | `wav.go` |
| `loader/config` | pass | Inventory/tests only; source review outstanding |
| `loader/gguf` | pass | `gguf.go`, `open_lifetime_linux_test.go` |
| `loader/gguf/llamaq4` | no test files | Inventory/tests only; source review outstanding |
| `loader/gguf/llamaq4plan9` | pass | Inventory/tests only; source review outstanding |
| `loader/numpy` | pass | `npz.go` |
| `loader/omnivoice` | pass | Inventory/tests only; source review outstanding |
| `loader/safetensors` | pass | `safetensors.go`, `resolve.go`, `audit_safety_test.go` |
| `loader/tokenizer` | pass | Inventory/tests only; source review outstanding |
| `loader/weights` | pass | Inventory/tests only; source review outstanding |
| `model` | pass | `gpu_forward.go`, `batch_prefill.go`, `frozen_gpu.go`, `frozen_gpu_prefix.go` |
| `model/bert` | pass | `bert.go` |
| `model/common` | no test files | Inventory/tests only; source review outstanding |
| `model/diffusiongemma` | pass | `expert_lru_cache.go` |
| `model/gemma` | pass | Inventory/tests only; source review outstanding |
| `model/gemma4` | no test files | Inventory/tests only; source review outstanding |
| `model/gliner2` | pass | Inventory/tests only; source review outstanding |
| `model/hunyuan3d` | pass | `runtime.go` |
| `model/ideogram4` | pass | Inventory/tests only; source review outstanding |
| `model/inspect` | pass | Inventory/tests only; source review outstanding |
| `model/internal/ops` | no test files | Inventory/tests only; source review outstanding |
| `model/internal/readiness` | no test files | Inventory/tests only; source review outstanding |
| `model/internal/tensorinspect` | no test files | Inventory/tests only; source review outstanding |
| `model/jevlike` | pass | `checkpoint.go`, `model.go`, `frozen_train.go` |
| `model/lfm2` | pass | Inventory/tests only; source review outstanding |
| `model/llama` | no test files | Inventory/tests only; source review outstanding |
| `model/minicpmv` | pass | `tensors.go` |
| `model/mosstranscribe` | pass | Inventory/tests only; source review outstanding |
| `model/omnivoice` | pass | `workers.go` |
| `model/qwen` | pass | Inventory/tests only; source review outstanding |
| `model/qwen3tts` | pass | Inventory/tests only; source review outstanding |
| `model/speaker` | pass | Inventory/tests only; source review outstanding |
| `model/speaker/community1` | pass | Inventory/tests only; source review outstanding |
| `model/trellis2` | pass | Inventory/tests only; source review outstanding |
| `model/whisper` | pass | `load_checked.go` |
| `runtime/expertstream` | pass | Inventory/tests only; source review outstanding |
| `runtime/graph` | pass | Inventory/tests only; source review outstanding |
| `runtime/inferencesched` | pass | `scheduler.go` |
| `runtime/kv` | pass | Inventory/tests only; source review outstanding |
| `runtime/memory` | pass | Inventory/tests only; source review outstanding |
| `runtime/promptcache` | pass | Inventory/tests only; source review outstanding |
| `runtime/quant` | pass | Inventory/tests only; source review outstanding |
| `runtime/resourcebudget` | pass | Inventory/tests only; source review outstanding |
| `runtime/sampling` | pass | Inventory/tests only; source review outstanding |
| `runtime/servingbench` | pass | Inventory/tests only; source review outstanding |
| `runtime/speechjob` | pass | `queue.go` |
| `runtime/speechjob/httpapi` | pass | `http.go` |
| `tensor` | pass | `shape.go`, `unsafe.go`, `tensor.go` |

## Specific CPU/loader/service boundary checks

* GGUF header count, size and tensor span checks were inspected; semantic-error descriptor cleanup was reproduced with a baseline overlay and fixed. NPZ metadata admission (16MiB archive, 8MiB expanded data, 16 members), literal-header parsing and exact-size checks were inspected; this was not fuzzing.
* BERT copies safetensors data into tensor-owned buffers before closing the source. Legacy Forward/Embed still rely on caller-valid token, sequence and attention-mask inputs; bad public input can panic, so this is not untrusted-service admission. Whisper's checked loader explicitly copies finite values and bounds metadata before materialisation; its legacy loaders are a separate coverage gap.
* OmniVoice borrowed GEMM pools require the owning backbone to outlive its siblings and prohibit concurrent lifecycle calls. Hunyuan3D TensorRef retains the source File and its converted tensors copy data, but concurrent Close/reads are not protected. MiniCPM-V inventory currently suppresses resolver errors into an unavailable flag; inspectors need a separate error-reporting policy.
* The inference scheduler serialises Step and Close but calls work methods while holding its mutex; work must not re-enter it or block indefinitely. MaxActive is not a bound on the waiting queue. Speech-job Shutdown waits for drain and Close refuses busy workers; the HTTP media decoder uses MaxBytesReader.
* The CUDA review covers every raw driver-call location through an AST inventory, not every higher-level buffer lifetime. Global capture ownership, scratch aliases and quiescent-only Shutdown remain open in issue #16.

## Outside the host package table

Bun: 20 script tests across eight files, 2,565 assertions. Python: six label-mass
and two parquet adapter tests. Documentation/link/layout checks pass. ARM64 and
RISC-V builds are compile-only. Other scripts, individual assembly kernels and
external integration services have inventory coverage, not exhaustive review.
The attempted independent CPU/model delegate timed out and adds no coverage.
