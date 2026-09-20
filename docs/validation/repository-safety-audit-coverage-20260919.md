## Repository safety audit: package coverage

Companion to [the audit report](repository-safety-audit-20260919.md).

This table began with 161 host-listed packages; the shared HTTP decoder adds a
162nd package. It is not a safety clearance. Selected sections in 75 packages
were inspected; **87 packages have inventory/test coverage only**. See the
[fourth-pass findings](repository-safety-fourth-pass-20260919.md) for current fixes
and open native/backend gaps.
"Boundary inspection" means selected functions/sections in the listed files were
read for ownership, bounds or lifecycle behaviour; it does **not** mean the whole
file or package was reviewed. Blank source coverage is explicit outstanding work.
The race column is the fourth-pass NVIDIA-disabled host sweep: 101 pass and 61
have no tests. Optional CGo/native implementations remain unexecuted even when
their host stubs build. A passing package may
contain skipped GPU, K3 or missing-fixture tests. Foreign-only files were not
executed. Test-source files explain regressions rather than broaden source coverage.

| Package | Host race result | Source boundary inspection (selected sections) |
|---|---|---|
| `backends/cuda/ptx` | pass | Inventory/tests only; source review outstanding |
| `backends/ggmlcompute` | no test files | `ggmlcompute.go` (native sections source-only; required libraries unavailable) |
| `backends/ggmlexec` | no test files | Inventory/tests only; source review outstanding |
| `backends/ggmlgraph` | no test files | `ggmlgraph.go`, `stub.go` (native sections source-only; required libraries unavailable) |
| `backends/ggmlquant` | no test files | `ggmlquant.go`, `stub.go` (native sections source-only; required libraries unavailable) |
| `backends/internal/ggmlutil` | pass | `ggmlutil.go`, `ggmlutil_test.go` |
| `backends/llamagraph` | no test files | Inventory/tests only; source review outstanding |
| `backends/mlx` | pass | `validate.go`, `load.go`, `gemv.go` |
| `backends/nvidia` | no test files | Inventory/tests only; source review outstanding |
| `backends/nvidia/internal/debuglog` | no test files | `debug.go` |
| `backends/nvidia/ioctl` | pass | `memory.go`, `gpfifo.go`, `ioctl.go` (source-only; no device calls) |
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
| `backends/placement` | pass | `placement.go` (selected sizing/placement sections) |
| `backends/simd` | no test files | Inventory/tests only; source review outstanding |
| `backends/simd/fft` | pass | `fft_simd.go`, `mel_fused.go` |
| `backends/simd/kernels` | pass | Inventory/tests only; source review outstanding |
| `backends/simd/quant/bf16` | pass | Inventory/tests only; source review outstanding |
| `backends/simd/quant/fp8` | pass | `fp8.go`, `batch.go`, `dot_amd64.go` |
| `backends/simd/quant/nvfp4` | pass | `validate.go` |
| `backends/simd/quant/q4` | pass | `validate.go` |
| `backends/simd/runtime` | pass | `gemm_parallel.go` |
| `backends/spacemit/aicpu` | no test files | Inventory/tests only; source review outstanding |
| `backends/spacemit/aicpu/aipool` | no test files | `ai_pool.go`, `ai_pool_new.go`, `ai_thread.go`, `ai_pool_lifecycle_riscv64_test.go` (cross-build only) |
| `backends/spacemit/aicpu/config` | no test files | Inventory/tests only; source review outstanding |
| `backends/spacemit/aicpu/q4kcshim` | no test files | Inventory/tests only; source review outstanding |
| `backends/spacemit/board` | pass | `backend.go`, `select.go`, `ops.go`, `simd.go`, `spacemit.go`, `vulkan.go` |
| `backends/spacemit/ime2` | pass | `worker_pool.go`, `worker_pool_test.go` |
| `backends/spacemit/inference` | pass | `inference.go`, `inference_test.go` |
| `backends/spacemit/ort` | pass | `spacemitort.go`, `options.go` (native sections source-only; required libraries unavailable) |
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
| `cmd/diffusiongemmaserver` | pass | `main.go` |
| `cmd/gliner2` | pass | Inventory/tests only; source review outstanding |
| `cmd/image/hy3dinspect` | no test files | Inventory/tests only; source review outstanding |
| `cmd/image/ideogram4gen` | no test files | Inventory/tests only; source review outstanding |
| `cmd/image/ideogram4inspect` | no test files | Inventory/tests only; source review outstanding |
| `cmd/image/ideogram4vaeprobe` | no test files | Inventory/tests only; source review outstanding |
| `cmd/image/ideogram4vaesmoke` | no test files | Inventory/tests only; source review outstanding |
| `cmd/image/internal/k3flags` | no test files | Inventory/tests only; source review outstanding |
| `cmd/image/zimageinspect` | no test files | Inventory/tests only; source review outstanding |
| `cmd/internal/dgflags` | no test files | Inventory/tests only; source review outstanding |
| `cmd/internal/testexec` | no test files | `helper.go` |
| `cmd/jevlike` | pass | `frozen.go` |
| `cmd/llm/internal/pathutil` | no test files | Inventory/tests only; source review outstanding |
| `cmd/llm/internal/promptfile` | no test files | Inventory/tests only; source review outstanding |
| `cmd/llm/llmchat` | no test files | Inventory/tests only; source review outstanding |
| `cmd/llm/llmgen` | pass | Inventory/tests only; source review outstanding |
| `cmd/llm/llmserver` | pass | `main.go` |
| `cmd/llm/servebench` | pass | `main.go`, `main_test.go` |
| `cmd/llm/specbench` | pass | Inventory/tests only; source review outstanding |
| `cmd/llm/speccheck` | pass | Inventory/tests only; source review outstanding |
| `cmd/minicpmvinspect` | no test files | `main.go` |
| `cmd/models/embcheck` | pass | `main.go`, `main_test.go` |
| `cmd/models/gemma4mtpparity` | pass | Inventory/tests only; source review outstanding |
| `cmd/models/gemma4mtpsmoke` | pass | Inventory/tests only; source review outstanding |
| `cmd/models/ggufinspect` | pass | Inventory/tests only; source review outstanding |
| `cmd/models/ggufsmoke` | pass | Inventory/tests only; source review outstanding |
| `cmd/models/lfm2inspect` | pass | `main.go`, `main_test.go` |
| `cmd/models/modelcoverage` | pass | Inventory/tests only; source review outstanding |
| `cmd/models/shapecheck` | pass | `main.go`, `main_test.go` |
| `cmd/qwen/qwen36run` | pass | Inventory/tests only; source review outstanding |
| `cmd/qwen/qwen3ttsinspect` | pass | `main.go`, `main_test.go` |
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
| `gpu` | pass | `attention_full.go`, `cross_attention.go`, `conv1d.go`, `attention_safety_test.go`, `conv1d_test.go` |
| `half` | pass | `half.go` |
| `internal/httpinput` | pass | `json.go`, `json_test.go` |
| `internal/checked` | no test files | `int.go` |
| `internal/floatcmp` | pass | `floatcmp.go`, `floatcmp_test.go` |
| `internal/ggmlfp16` | pass | `gelu.go`, `gelu_amd64.go`, `gelu_other.go` |
| `internal/modelcoverage` | no test files | Inventory/tests only; source review outstanding |
| `loader/audio` | pass | `wav.go`, `wav_invalid_test.go` |
| `loader/audio/media` | pass | `wav.go` |
| `loader/config` | pass | `config.go`, `fixture_tensor_summary.go`, `fixture_tensor_summary_test.go` |
| `loader/gguf` | pass | `gguf.go`, `open_lifetime_linux_test.go` |
| `loader/gguf/llamaq4` | no test files | Inventory/tests only; source review outstanding |
| `loader/gguf/llamaq4plan9` | pass | Inventory/tests only; source review outstanding |
| `loader/numpy` | pass | `npz.go` |
| `loader/omnivoice` | pass | `layer_buffer.go` |
| `loader/safetensors` | pass | `safetensors.go`, `resolve.go`, `audit_safety_test.go` (copy/close and borrowed raw contracts) |
| `loader/tokenizer` | pass | Inventory/tests only; source review outstanding |
| `loader/weights` | pass | `weights.go`, `weights_test.go` |
| `model` | pass | `gpu_forward.go`, `batch_prefill.go`, `frozen_gpu.go`, `frozen_gpu_prefix.go`, `mtp_prompt_context.go`, `mtp_verifier_forward_test.go`, `rope.go`, `ggml_flash_ref.go` (selected sections) |
| `model/bert` | pass | `bert.go` |
| `model/common` | pass | `config.go`, `config_test.go` |
| `model/diffusiongemma` | pass | `expert_lru_cache.go` |
| `model/gemma` | pass | `config.go`, `config_test.go` |
| `model/gemma4` | no test files | `compat.go`, `doc.go` |
| `model/gliner2` | pass | Inventory/tests only; source review outstanding |
| `model/hunyuan3d` | pass | `runtime.go` |
| `model/ideogram4` | pass | `fp8_load.go`, `fp8_linear.go` |
| `model/inspect` | pass | `predicates.go`, `predicates_test.go` |
| `model/internal/ops` | pass | `ops.go`, `ops_test.go` |
| `model/internal/readiness` | pass | `report.go`, `report_test.go` |
| `model/internal/tensorinspect` | pass | `shape.go`, `tensors.go`, `tensors_test.go` |
| `model/jevlike` | pass | `checkpoint.go`, `model.go`, `frozen_train.go` |
| `model/lfm2` | pass | `config.go`, `state.go`, `runtime_request.go`, `context.go`, `conv_state.go`, `attention_kv.go`, `embedding_layout.go`, `router_layout.go`, `tensor_shape_validation.go`, `sizing.go`, stage/layout validators and schedule membership checks |
| `model/llama` | no test files | `rope.go` |
| `model/minicpmv` | pass | `tensors.go` |
| `model/mosstranscribe` | pass | `load.go`, `native.go`, `audio.go`, `native_test.go` |
| `model/omnivoice` | pass | `workers.go` |
| `model/qwen` | pass | Inventory/tests only; source review outstanding |
| `model/qwen3tts` | pass | Inventory/tests only; source review outstanding |
| `model/speaker` | pass | Inventory/tests only; source review outstanding |
| `model/speaker/community1` | pass | Inventory/tests only; source review outstanding |
| `model/trellis2` | pass | Inventory/tests only; source review outstanding |
| `model/whisper` | pass | `load_checked.go` |
| `runtime/expertstream` | pass | `reader.go`, `manifest.go`, `alloc.go`, `types.go`, `lifetime_test.go` |
| `runtime/graph` | pass | `graph.go`, `plan.go`, `executor.go`, `safety_test.go` |
| `runtime/inferencesched` | pass | `scheduler.go` |
| `runtime/kv` | pass | Inventory/tests only; source review outstanding |
| `runtime/memory` | pass | Inventory/tests only; source review outstanding |
| `runtime/promptcache` | pass | `cache.go`, `identity.go` |
| `runtime/quant` | pass | `gptq.go`, `mlx.go`, `nvfp4.go`, `gemv_q4.go`, `gemv_q4_validate.go`, `gptq_validate.go` |
| `runtime/resourcebudget` | pass | `budget.go` |
| `runtime/sampling` | pass | `sampling.go`, `allocation_test.go` |
| `runtime/servingbench` | pass | `sse.go`, `servingbench.go`, `arrival.go`, `summary.go` |
| `runtime/speechjob` | pass | `queue.go` |
| `runtime/speechjob/httpapi` | pass | `http.go` |
| `tensor` | pass | `shape.go`, `unsafe.go`, `tensor.go`, `conv1d.go`, `fuse.go` |

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
