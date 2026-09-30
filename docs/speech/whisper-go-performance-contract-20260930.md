# Native Go Whisper performance contract

Native Go Whisper must match or beat the retained whisper.cpp pipeline on Sigma, with Vulkan, flash attention and genuine speech VAD. This is an active implementation and qualification task. Historical alternative-backend timings establish feasibility on the hardware; they do not establish native Go acceptance.

## Baseline and isolation

- Hardware: Intel i5-1340P, Iris Xe RPL-P (`8086:a7a0`), 32 GB shared RAM; AVX2/FMA/AVX-VNNI. No AVX-512, AMX or XMX.
- Original engine: whisper.cpp `c44b60b8053bbf2a5c1e014f11323fb3f2485177`, including the local original-timeline VAD-token export correction.
- Original selected model: multilingual large-v3-turbo Q5_0, SHA-256 `394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2`.
- Original VAD: Silero 6.2.0, SHA-256 `2aa269b785eeb53a82983a20501ddf7c1d9c48e33ab63a41391ac6c9f7fb6987`.
- Reference options: four CPU threads, affinity CPUs 0–7, Intel Vulkan, flash attention, greedy beam/best-of 1, explicit fixture language. Record temperature fallback, timestamp and VAD settings rather than assuming defaults match.
- Candidate: native Go model orchestration and CPU/SIMD/Vulkan kernels. Calling the original engine through a Go wrapper does not meet the objective.
- Native comparison runs use the same admitted CPU/memory/device limits, with process-local zero swap, live-Qwen-idle and host-available-memory guards. Keep production and stopped legacy services unchanged.

The currently authorised Nemotron Decoder pipeline must drain before a new Whisper compute window. Source review, metadata inspection and documentation can proceed without competing inference. Its retained plain exports are independent of this goal.

## Matched measurements

Pin both binaries, model revisions/conversions, tensor precision, driver, options and audio inputs before changing the candidate. Preserve baseline binaries and raw evidence outside mutable result paths. Existing Go HF F32 weights and the original Q5 model have different numerical representations; that difference must be explicit.

| Comparison | Purpose |
|---|---|
| Go CPU versus Go Vulkan, VAD off, same precision/settings | Locate encoder/decoder costs without mixing VAD or quantisation |
| Original versus Go Vulkan, VAD off | Establish whole-model and backend gap |
| Original versus Go Vulkan, VAD on with equivalent speech spans/options | Target full pipeline and original-timeline mapping |
| Word timestamps off/on in both engines | Separate alignment cost and validate functionality |
| Cold versus warm loading/preparation/request | Expose costs moved into preparation or retained memory |
| F32 versus explicitly named packed/quantised candidates | Measure numerical/quality and memory trade-offs |

Use the retained JFK, Portuguese/French, silence-padded and silence-only fixtures, then representative Decoder crops and the full Decoder recording. Repeat small arms at least five times, preferably ten for noisy differences; alternate order and compare medians/distributions. Do not accept the best single run. Full-length time is an additional gate and must not be extrapolated from a short fixture.

Report media decode, frontend, VAD, encoder, cross-KV preparation, token generation, word alignment, publication and cleanup separately where observable. Count setup/teardown consistently in the primary whole-ASR measurement. Model loading is reported separately as well as in cold total time. Record application transfer/wait/allocation counters, peak/retained memory and process swap; driver-internal memory movement is not implied by application counters.

The acceptance target is a Go median whole-ASR latency no greater than the original under comparable settings and output features, without an unexplained quality or memory regression. Missing native platforms, annotation or baseline coverage stay open.

## Implementation inventory

### SIMD

The decoder already uses preallocated scratch, self-KV, precomputed cross-KV, `Sdot`, `Sdotx4`, `Saxpy` and SIMD layer normalisation. Initial candidates are fused output-row projections, value accumulation with preserved skip/reduction rules, bounded workspace reuse and immutable cross-KV sharing for the separate alignment pass.

Relevant sources: [decoder buffers](../../model/whisper/decoder_bufs.go), [decoder attention](../../model/whisper/decoder.go), [linear dispatch](../../model/whisper/linear_opt.go), [word alignment](../../model/whisper/word_alignment.go), and [generic row-dot kernels](../../backends/simd/runtime/dotrowsx4.go).

A generic kernel is not automatically exact for Whisper. Validate vector reductions, tails, aliasing, nonfinite values, masking and the attention weight skip threshold before substitution. Existing `DotRowsx4` tests allow a numerical tolerance; they do not establish bitwise equivalence on every supported input.

The [current Whisper INT8 implementation](../../model/whisper/int8_linear.go) is RISC-V-only; the [other-architecture implementation](../../model/whisper/int8_linear_other.go) disables it. An amd64 VNNI or packed-weight backend requires explicit implementation and calibration. Exact SIMD and allocation changes are evaluated before quantisation; no tolerance widening or default precision change is implied.

### Vulkan and flash attention

The [resident encoder](../../model/whisper/vulkan_encoder.go) already keeps intermediate tensors on the device. [VkAttentionF32](../../backends/vulkan/vulkan_attention.go) and its [shader](../../backends/vulkan/shaders/attention_f32.glsl) use 16-query/16-key tiles, online max/sum softmax and no full score matrix. The admitted non-causal geometry covers the turbo encoder's 20 heads of width 64 and 1500 time positions.

This implementation needs phase profiling, kernel/submission tuning and native parity/performance gates. It does not provide causal or arbitrary masked decoder attention. A decoder GPU path needs its own checked contract if measured costs justify it. Avoid many small host/device transitions; maintain native owner, barrier, fence, cancellation and drain rules. No silent CPU fallback may conceal a failed native measurement.

### Genuine VAD

The checked [PCM transcription path](../../model/whisper/pcm_transcribe.go) only skips exact digital zero. Existing energy-threshold speaker VAD does not establish Silero-equivalent speech probabilities.

The retained Silero checkpoint is a legacy GGML file, not GGUF. Metadata and full-file hash were inspected without running inference:

| Field | Value |
|---|---|
| File bytes | 885,098 |
| Type/version | `silero-16k`, 6.2.0 |
| Window/context | 512 / 64 samples |
| Encoder | Four kernel-3 layers: 129→128→64→64→128 |
| LSTM input/hidden | 128 / 128 |
| Final input/output | 128 / 1 |
| Tensor count | 15 |
| Storage | F16 convolution/STFT weights; F32 remaining tensors, including a rank-zero scalar |

Implement bounded parsing/conversion under the loader, model-specific state/orchestration under a native VAD model package, and reusable numerical kernels under the backend. Pin source/model identities and verify shapes, offsets, dtype, size arithmetic, finite values and closure. No dependency on a developer cache or download belongs in ordinary unit tests.

The reference graph uses reflect-padded STFT magnitude, four convolution/ReLU layers, LSTM state and final ReLU/convolution/sigmoid. Test frame probabilities against an independent pinned reference; fresh recording/retry/reset must not inherit recurrent state. Then qualify thresholding, speech duration/padding and original-timeline mapping. Match reference behaviour at boundaries before integrating PCM compaction or segment scheduling.

VAD gates include leading/trailing silence, quiet speech, short islands, long gaps, overlap, complete silence, tail padding, cancellation, and restored segment/word times. Do not invent timestamps in removed regions or classify every low-energy interval as silence. Preserve verified transcript checkpoints and exports on failure.

## Correctness and completion

- Retain fixed scalar/upstream references and calibrated tolerances. Exact arithmetic changes preserve their contracts; quantised changes use separately named modes and explicit quality evaluation.
- Verify generated tokens, segment coverage, word timing, language policy, cancellation/fresh retry, reset/reuse and output ownership. Hash checks establish asset/export integrity, not recognition accuracy.
- Use independently annotated audio samples for acoustic WER and timing/speaker questions. Published edited Decoder text remains a discrepancy diagnostic; its 21.49% difference is not acoustic WER.
- Run affected model-free tests repeatedly and under race detection, whole-tree tests/vet/builds, layout/docs checks, relevant cross-builds and authorised native gates. Ordinary tests must remain small, deterministic and offline.
- Preserve the active service, profiles, model defaults and job state. Deploy only after passing evidence and separate idle-service clearance.
- Commit and push validated changes and bounded evidence. Mark the goal complete only after Vulkan, flash attention, genuine VAD, matched speed and required quality/timing gates have passed.

Earlier feasibility analysis: [Whisper speed/SIMD assessment](../validation/whisper-sigma-speed-assessment-20260930.md). As of this contract, no new native Go Whisper optimisation or VAD inference is performance-qualified.
