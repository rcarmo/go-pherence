## Repository safety audit: package coverage

Companion to [the audit report](repository-safety-audit-20260919.md).

This table began with 161 host-listed packages; shared HTTP decoding, command
capture and packed-Q4 validation bring it to 164. All **164 packages** now have
selected source-boundary review; **zero are inventory-only**. This is not a safety
clearance or full-file semantic review.
See the [closeout findings](repository-safety-audit-closeout-20260920.md) for current fixes
and open native/backend gaps.
"Boundary inspection" means selected functions/sections in the listed files were
read for ownership, bounds or lifecycle behaviour; it does **not** mean the whole
file or package was reviewed. Native and deeper semantic gaps remain in the [open-findings ledger](repository-safety-audit-open-findings-20260920.md); package coverage is not remediation completion.
The race column is the post-audit host-remediation NVIDIA-disabled sweep: 110 pass and 54
have no tests. Optional CGo/native implementations remain unexecuted even when
their host stubs build. A passing package may
contain skipped GPU, K3 or missing-fixture tests. Foreign-only files were not
executed. Test-source files explain regressions rather than broaden source coverage.

| Package | Host race result | Source boundary inspection (selected sections) |
|---|---|---|
| `backends/cuda/ptx` | pass | `attention_full.go`, `conv1d.go`, `fft.go`, `pool.go`, `encoder_ops.go`: signatures/index geometry; host Whisper preflight fixed, PTX execution unavailable |
| `backends/ggmlcompute` | no test files | `ggmlcompute.go` (native sections source-only; required libraries unavailable) |
| `backends/ggmlexec` | no test files | `ggmlexec.go` (capability/island planner; no native execution) |
| `backends/ggmlgraph` | no test files | `ggmlgraph.go`, `stub.go` (native sections source-only; required libraries unavailable) |
| `backends/ggmlquant` | no test files | `ggmlquant.go`, `stub.go` (native sections source-only; required libraries unavailable) |
| `backends/internal/ggmlutil` | pass | `ggmlutil.go`, `ggmlutil_test.go` |
| `backends/llamagraph` | pass | `llamagraph.go`, `stub.go`, shared `config.go`/`config_validation.go`, selected native `csrc/llamagraph.c` init/setter/cleanup (host config tests; native lifecycle/dtype/upload gaps open) |
| `backends/mlx` | pass | `validate.go`, `load.go`, `gemv.go` |
| `backends/nvidia` | no test files | `doc.go` (namespace only) |
| `backends/nvidia/internal/debuglog` | no test files | `debug.go` |
| `backends/nvidia/ioctl` | pass | `ioctl.go`, `files.go`, `memory.go`, `gpfifo.go` (fake descriptor tests; no device calls) |
| `backends/nvidia/ptx` | no test files | `attention.go`, `conversion.go`, `norm.go`, `sgemm_compensated.go` |
| `backends/nvidia/ptx/bf16` | no test files | `bf16.go`, `native.go`: norm/GEMV block/shared extents and narrowing (delegated source review; native execution unavailable) |
| `backends/nvidia/ptx/fp8` | no test files | `fp8.go`: GEMV/GEMM/dequant indexing and wrapper contracts (delegated; u32 products open) |
| `backends/nvidia/ptx/ideogram` | no test files | `ideogram.go`: norm/adaLN/RoPE/attention signatures and indexing (delegated source-only) |
| `backends/nvidia/ptx/mlx` | no test files | `mlx.go`, `selected_expert.go`: GEMM/persistent-work/pointer lifetime contracts (delegated; product bounds open) |
| `backends/nvidia/ptx/nvfp4` | no test files | `nvfp4.go`: dequant/GEMV groups and extents (delegated; GEMV product bounds open) |
| `backends/nvidia/ptx/q4` | no test files | `gemv_q4k.go`, `gate_up_gelu_q4k_work.go`, `gemm.go`: launch/reduction/group contracts (delegated; batched u32 products open) |
| `backends/nvidia/ptx/q5` | no test files | `q5.go`, `q5_scatter_work.go`, `q5_scatter_work_ptrs.go`: batching/scatter/indexing (delegated; product bounds open) |
| `backends/nvidia/ptx/q8` | no test files | `q8.go`, `q8_batch.go`, `q8_scatter.go`, `q8_scatter_work.go`, `q8_scatter_work_ptrs.go`: scatter/indexing; duplicate-position race remains source finding |
| `backends/nvidia/runtime` | pass | `whisper.go`, `whisper_bounds.go`, `driver_scope.go`, `driver_call_inventory_test.go`, `driver_scope_test.go`, `context_boundary_test.go`, `runtime.go`, `streams.go`, `compiler.go`, `compiler_test.go`, `devbuf.go`, `argmax.go`, `bf16_native.go`, `mega_module.go`, `module_state.go`, `attention_splitkv.go`, `bf16_projection.go` |
| `backends/placement` | pass | `placement.go` (selected sizing/placement sections) |
| `backends/simd` | no test files | `doc.go` (namespace only) |
| `backends/simd/fft` | pass | `fft_simd.go`, `mel_fused.go` |
| `backends/simd/kernels` | pass | `activation.go`, `attention.go`, `rope.go`, `shape.go`, `softmax.go`, `layernorm.go`, `gelu.go` |
| `backends/simd/quant/bf16` | pass | `bf16.go`, `checked.go`, nonfinite/rounding regressions |
| `backends/simd/quant/fp8` | pass | `fp8.go`, `batch.go`, `dot_amd64.go` |
| `backends/simd/quant/nvfp4` | pass | `validate.go` |
| `backends/simd/quant/q4` | pass | `validate.go` |
| `backends/simd/runtime` | pass | `gemm_parallel.go`, `rope.go` (selected dispatch/fallback boundary) |
| `backends/spacemit/aicpu` | no test files | `main.go`, `parallel_decode_ai.go`, `q4k_llama_x32.go`, `q4k_ai.go`, `q4k_m4_dispatch.go`, `q4k_tcm_bwave.go`, `tcm_parallel.go`, C-shim native/stub and pooled dispatch (delegated source-only; ownership/extent gaps open) |
| `backends/spacemit/aicpu/aipool` | no test files | `ai_pool.go`, `ai_pool_new.go`, `ai_thread.go`, `ai_pool_lifecycle_riscv64_test.go` (cross-build only) |
| `backends/spacemit/aicpu/config` | no test files | `flags.go` (initialisation and mutable configuration contract) |
| `backends/spacemit/aicpu/q4kcshim` | no test files | `q4kcshim.go`, `q4kcshim_stub.go` (native buffer/narrowing gaps; source-only) |
| `backends/spacemit/board` | pass | `backend.go`, `select.go`, `ops.go`, `simd.go`, `spacemit.go`, `vulkan.go` |
| `backends/spacemit/ime2` | pass | `worker_pool.go`, `worker_pool_test.go` |
| `backends/spacemit/inference` | pass | `inference.go`, `inference_test.go` |
| `backends/spacemit/ort` | pass | `spacemitort.go`, `options.go` (native sections source-only; required libraries unavailable) |
| `backends/spacemit/rvv` | pass | `f16.go`, `fallback_other.go`, `packed_outer_common.go`, GEMM/quant/W4 wrappers, `copy_rvv.go`, `f16_convert.go` (delegated; ISA/extent/tail contracts open) |
| `backends/spacemit/tcm` | pass | `tcm.go`, `tcm_stub.go` (borrowed-pointer/Close gaps; native source-only) |
| `backends/vulkan` | pass | `vulkan_init.go`, `vulkan_ops.go`, `vulkan_wrapper_test.go`, `vulkan_buf.go` |
| `cmd/audio/diarize-vtt` | pass | `main.go`: flags, model/audio setup, chunk construction and materialisation; whole-audio/subprocess admission and nonfinite knobs open |
| `cmd/audio/internal/whisperflags` | pass | `whisperflags.go` (process environment policy) |
| `cmd/audio/moss-transcribe` | pass | `main.go`: run/flags, native load/Close, WAV/prompt/generation/output paths; trusted-local workload |
| `cmd/audio/omnivoice` | pass | `main.go`, `serve.go`: flag admission, NDJSON bounds, phrase cache, chunk/file ownership; sequential local worker, not network server |
| `cmd/audio/speakercheck` | pass | `main.go`: waveform slicing, ffmpeg fallback, expected-label scoring and output; private temp/finite offsets/complete score fixes tested; whole-audio/size admission open |
| `cmd/audio/speechjob` | pass | `main.go`, `client.go`: endpoint/token/deadline policy, response bounds, authenticated artifact hashing and no-clobber publication |
| `cmd/audio/speechjobserve` | pass | `config.go`, `server.go`, `server_start_linux_amd64.go`, startup/profile stubs: bounded asset/config policy, connection admission, drain/quarantine ownership (native execution unverified) |
| `cmd/audio/whisper` | pass | `main.go`: flags, loader/GPU choice, materialisation/chunking/output (legacy whole-audio and subprocess admission gaps) |
| `cmd/audio/whisperffndiag` | no test files | `main.go`: load/resample/mel/probe setup (delegated; whole-audio diagnostic bounds open) |
| `cmd/diffusiongemmainspect` | no test files | `main.go`: metadata/shard/readiness gates, optional weight owner and residency estimates; nonfinite GiB conversion open |
| `cmd/diffusiongemmarun` | pass | `main.go`: flags/environment, prompt framing, GGUF open/prewarm/fatal-cleanup and profile ownership (selected sections; native lifetime remains unverified) |
| `cmd/diffusiongemmaserve` | pass | `main.go`: HTTP framing/admission/token/transport limits and model lock; host-only handler tests, native preemption/auth open |
| `cmd/diffusiongemmaserver` | pass | `main.go` |
| `cmd/gliner2` | pass | `main.go`: task compatibility, schema parser, budget/threshold checks and owned model scoring; aggregate local schema/text admission open |
| `cmd/image/hy3dinspect` | no test files | `main.go`: input/setup/load/output ownership (delegated; local allocation caps remain except bounded/closed VAE smoke fix) |
| `cmd/image/ideogram4gen` | no test files | `main.go`: input/setup/load/output ownership (delegated; local allocation caps remain except bounded/closed VAE smoke fix) |
| `cmd/image/ideogram4inspect` | no test files | `main.go`: input/setup/load/output ownership (delegated; local allocation caps remain except bounded/closed VAE smoke fix) |
| `cmd/image/ideogram4vaeprobe` | no test files | `main.go`: input/setup/load/output ownership (delegated; local allocation caps remain except bounded/closed VAE smoke fix) |
| `cmd/image/ideogram4vaesmoke` | pass | `main.go`: input/setup/load/output ownership (delegated; local allocation caps remain except bounded/closed VAE smoke fix) |
| `cmd/image/internal/k3flags` | no test files | `k3flags.go` (process environment policy) |
| `cmd/image/zimageinspect` | no test files | `main.go`: input/setup/load/output ownership (delegated; local allocation caps remain except bounded/closed VAE smoke fix) |
| `cmd/internal/dgflags` | pass | `dgflags.go`, budget-conversion regressions |
| `cmd/internal/testexec` | no test files | `helper.go` |
| `cmd/jevlike` | pass | `frozen.go` |
| `cmd/llm/internal/pathutil` | no test files | `base.go` |
| `cmd/llm/internal/promptfile` | no test files | `promptfile.go` (aggregate admission remains caller-owned) |
| `cmd/llm/llmchat` | no test files | `main.go`: flag/REPL/model/generation boundaries (delegated; local output/work caps open) |
| `cmd/llm/llmgen` | pass | `main.go`: generation accounting and MTP sizing (delegated plus direct); arithmetic fixed, prepared-prompt accounting follow-up |
| `cmd/llm/llmserver` | pass | `main.go` |
| `cmd/llm/servebench` | pass | `main.go`, `main_test.go` |
| `cmd/llm/specbench` | pass | `main.go`: flags/model/prompt loops/CSV (delegated; no new selected-boundary finding) |
| `cmd/llm/speccheck` | pass | `main.go`: golden parse/write and compare (delegated plus direct); bounded single JSON fixed |
| `cmd/minicpmvinspect` | no test files | `main.go` |
| `cmd/models/embcheck` | pass | `main.go`, `main_test.go` |
| `cmd/models/gemma4mtpparity` | pass | `main.go`: fixture-only versus execution, probe validation and strict reporting; false parity fixed; host-only fixtures |
| `cmd/models/gemma4mtpsmoke` | pass | `main.go`: flags, load, checked KV allocation and step reporting (delegated) |
| `cmd/models/ggufinspect` | pass | `main.go`: metadata/expectation/readiness gates; expectation-only plan admission fixed |
| `cmd/models/ggufsmoke` | pass | `main.go`: static/runtime expectations, prompt helpers and generation paths; runtime planner failure no longer ignored |
| `cmd/models/lfm2inspect` | pass | `main.go`, `main_test.go` |
| `cmd/models/modelcoverage` | pass | `main.go`: manifest/summarise/filter/roadmap; empty family rejection fixed; category/filtered scope distinction retained |
| `cmd/models/shapecheck` | pass | `main.go`, `main_test.go` |
| `cmd/qwen/qwen36run` | pass | `main.go` flag/setup/owner/cache/prefill sections, `mlx_lmhead.go` upload/GEMV/readback; cache conversion/work/sync findings open |
| `cmd/qwen/qwen3ttsinspect` | pass | `main.go`, `main_test.go` |
| `cmd/qwen/qwenmtpmeta` | pass | `main.go`: resolver errors, metadata enumeration and report claims fixed; no load/execution parity implied |
| `cmd/qwen/qwenmtpsmoke` | pass | `main.go`: metadata/load/synthetic input/forward (delegated; local dimension admission open) |
| `cmd/qwen/qwenmtpsynth` | pass | `main.go`: steps/draft/plan output (delegated; local maximum-step admission open) |
| `cmd/spacemit/ime2run` | no test files | `main.go`, `main_stub.go`: CLI sizing/error/owner/hardware side effects (delegated source review; native execution unavailable; dispositions in open-findings ledger) |
| `cmd/spacemit/ime2test` | no test files | `main.go`, `main_stub.go`: CLI sizing/error/owner/hardware side effects (delegated source review; native execution unavailable; dispositions in open-findings ledger) |
| `cmd/spacemit/npu-tcm` | no test files | `main.go`, `main_stub.go`: CLI sizing/error/owner/hardware side effects (delegated source review; native execution unavailable; dispositions in open-findings ledger) |
| `cmd/spacemit/spacemit_bench` | no test files | `main.go`: CLI sizing/error/owner/hardware side effects (delegated source review; native execution unavailable; dispositions in open-findings ledger) |
| `cmd/spacemit/spacemit_ffnblockbench` | no test files | `main.go`: CLI sizing/error/owner/hardware side effects (delegated source review; native execution unavailable; dispositions in open-findings ledger) |
| `cmd/spacemit/spacemit_ggmlbench` | no test files | `main.go`: CLI sizing/error/owner/hardware side effects (delegated source review; native execution unavailable; dispositions in open-findings ledger) |
| `cmd/spacemit/spacemit_ggmlplan` | no test files | `main.go`: CLI sizing/error/owner/hardware side effects (delegated source review; native execution unavailable; dispositions in open-findings ledger) |
| `cmd/spacemit/spacemit_graphfusebench` | no test files | `main.go`: CLI sizing/error/owner/hardware side effects (delegated source review; native execution unavailable; dispositions in open-findings ledger) |
| `cmd/spacemit/spacemit_graphrun` | no test files | `main.go`, `main_stub.go`: CLI sizing/error/owner/hardware side effects (delegated source review; native execution unavailable; dispositions in open-findings ledger) |
| `cmd/spacemit/spacemit_llama` | no test files | `main.go`, `main_stub.go`: CLI sizing/error/owner/hardware side effects (delegated source review; native execution unavailable; dispositions in open-findings ledger) |
| `cmd/spacemit/spacemit_ortbench` | pass | `main.go`: CLI sizing/error/owner/hardware side effects (delegated source review; native execution unavailable; dispositions in open-findings ledger) |
| `cmd/spacemit/spacemit_ortlayerbench` | no test files | `main.go`: CLI sizing/error/owner/hardware side effects (delegated source review; native execution unavailable; dispositions in open-findings ledger) |
| `cmd/spacemit/spacemit_plandump` | no test files | `main.go`: CLI sizing/error/owner/hardware side effects (delegated source review; native execution unavailable; dispositions in open-findings ledger) |
| `cmd/spacemit/spacemit_qbench` | no test files | `main.go`: CLI sizing/error/owner/hardware side effects (delegated source review; native execution unavailable; dispositions in open-findings ledger) |
| `cmd/spacemit/spacemit_run` | no test files | `main.go`: CLI sizing/error/owner/hardware side effects (delegated source review; native execution unavailable; dispositions in open-findings ledger) |
| `cmd/spacemit/testi8i4` | no test files | `main.go`, `main_stub.go`: CLI sizing/error/owner/hardware side effects (delegated source review; native execution unavailable; dispositions in open-findings ledger) |
| `cmd/spacemit/verifydot` | no test files | `main.go`, `main_stub.go`: CLI sizing/error/owner/hardware side effects (delegated source review; native execution unavailable; dispositions in open-findings ledger) |
| `cmd/tinydemo` | no test files | `main.go`: fixed small tensor demo (delegated; no external input) |
| `docs` | pass | `model_layout_test.go`, `model_coverage_manifest_test.go`: worktree exclusions/path guards, metadata-to-file assertions; test-only package, not runtime validation |
| `gpu` | pass | `attention_full.go`, `cross_attention.go`, `conv1d.go`, `attention_safety_test.go`, `conv1d_test.go` |
| `half` | pass | `half.go`, FP16/BF16 NaN classification/finite rounding regressions |
| `internal/httpinput` | pass | `json.go`, `json_test.go` |
| `internal/commandcapture` | pass | `capture.go`, `process_linux.go`, `process_other.go`, fake helper regressions |
| `internal/checked` | no test files | `int.go` |
| `internal/floatcmp` | pass | `floatcmp.go`, `floatcmp_test.go` |
| `internal/ggmlfp16` | pass | `gelu.go`, `gelu_amd64.go`, `gelu_other.go` |
| `internal/modelcoverage` | no test files | `manifest.go`: shared schema definitions only (delegated and direct) |
| `loader/audio` | pass | `wav.go`, `resample.go`, bounds regressions |
| `loader/audio/media` | pass | `wav.go` |
| `loader/config` | pass | `config.go`, `fixture_tensor_summary.go`, `fixture_tensor_summary_test.go` |
| `loader/gguf` | pass | `gguf.go`, `tokenizer.go`, lifetime/token bounds regressions |
| `loader/gguf/internal/q4layout` | pass | `layout.go`, overflow/narrowing/tail tests |
| `loader/gguf/llamaq4` | pass | `kernel_cgo_amd64.go`, `kernel_nocgo.go`, selected `kernel_amd64.c` orchestration (CPU synthetic tests; not exhaustive native audit) |
| `loader/gguf/llamaq4plan9` | pass | `kernel_amd64.go`, `kernel_other.go` (bounds/dispatch and stub parity; assembly internals not exhaustively reviewed) |
| `loader/numpy` | pass | `npz.go` |
| `loader/omnivoice` | pass | `layer_buffer.go`, `wav.go` |
| `loader/safetensors` | pass | `safetensors.go`, `resolve.go`, `audit_safety_test.go` (copy/close and borrowed raw contracts) |
| `loader/tokenizer` | pass | `tokenizer.go`, `sidecar.go` (source-only; frozen files unchanged; work-budget/collision/fallback semantics open) |
| `loader/weights` | pass | `weights.go`, `weights_test.go` |
| `model` | pass | `gpu_forward.go`, `batch_prefill.go`, `frozen_gpu.go`, `frozen_gpu_prefix.go`, `mtp_prompt_context.go`, `mtp_verifier_forward_test.go`, `rope.go`, `ggml_flash_ref.go` (selected sections) |
| `model/bert` | pass | `bert.go` |
| `model/common` | pass | `config.go`, `config_test.go` |
| `model/diffusiongemma` | pass | `expert_lru_cache.go` |
| `model/gemma` | pass | `config.go`, `config_test.go` |
| `model/gemma4` | no test files | `compat.go`, `doc.go` |
| `model/gliner2` | pass | `model.go`, `weights.go`, `deberta.go`, `encoding.go`, `heads.go`, `config.go`, `shared_scorer.go` (selected loading/validation sections; other inference/decoder paths outstanding) |
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
| `model/omnivoice` | pass | `workers.go`, `backend.go` (probe subprocess boundary) |
| `model/qwen` | pass | `prompt_cache.go`, `schedule.go`, `qwen35_source.go`, `qwen35_load_helpers.go`, `qwen35_validate_helpers.go` (selected loading/key/planning sections; CPU sidecar admission/locking fixed; GPU lifecycle validation remains) |
| `model/qwen3tts` | pass | `config.go`, `config_numbers.go`, `sizing.go`, embedding/attention/FFN/prefill/input layouts, frame/decoder/speaker/request sizing, `shapes.go`, tensor shape and stage-contract validation (metadata only; execution not implemented) |
| `model/speaker` | pass | `config.go`, `load.go`, `embed.go`, `diarize.go`, selected `ecapa.go` construction: copied weights and legacy unchecked inference/VAD/clustering inputs; findings open |
| `model/speaker/community1` | pass | `diarization_pcm.go`, `load_segmentation.go`, `load_resnet.go`, `vulkan_diarization.go`: bounds before payload/PCM allocation, copied ownership and native owner gate (selected sections only; device/numerical qualification open) |
| `model/trellis2` | pass | `sparse.go`, wrapped sizing/forged row/projection tests (sparse primitive, not full pipeline) |
| `model/whisper` | pass | `load_checked.go` |
| `runtime/expertstream` | pass | `reader.go`, `manifest.go`, `alloc.go`, `types.go`, `lifetime_test.go` |
| `runtime/graph` | pass | `graph.go`, `plan.go`, `executor.go`, `safety_test.go` |
| `runtime/inferencesched` | pass | `scheduler.go` |
| `runtime/kv` | pass | `cache.go`, `layered_f32_store.go`, selected `turboquant.go` constructor/ownership sections; storage/reset and `reuse.go` metadata/pre-clone/concurrency regressions (not full quantisation review) |
| `runtime/memory` | pass | `mmap_advisor.go`, Detach/serialised eviction/overlap-union tests; independent raw mappings still caller-owned |
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

Bun: 22 script tests across nine files (assertion count varies with document inventory). Python: six label-mass
and two parquet adapter tests from the first pass (not rerun here). Documentation/link/layout checks pass. ARM64 and
RISC-V builds are compile-only. Other scripts, individual assembly kernels and
external integration services have inventory coverage, not exhaustive review.
Earlier CPU/model and Qwen3-TTS delegates timed out and add no coverage. Eighth-pass completed delegated reviews covered the image/small CLI, model diagnostics, SpacemiT CLI, PTX interface and native-wrapper groups. Two larger command delegates timed out; their selected entry boundaries were read directly, with no credit for unfinished delegate work.
