# Grouped original-Q5 projection and structural trials — 1 October 2026

A standalone native Vulkan Q5_0 projection now preserves original stored values while reducing large FFN kernel time. It is an explicit operator only: Whisper does not yet consume its packed storage, and no end-to-end speed improvement is established. User feedback redirected this window towards execution-level changes; batching and projection/GELU fusion were also tested and rejected.

## Packed storage and arithmetic

`NewVkLinearQ5GroupedF32` accepts row-major original GGML Q5_0 blocks, 22 bytes for 32 values. It copies each scale, high-bit word and four quant words into six aligned uint32 words (24 bytes/block). Two padding bytes are zero. There is no re-quantisation. Eight lanes share a block's decoding, each writing two lower and two upper weights into a 64×32 F32 shared tile. The normal 64×64 output tile and increasing-K projection order are unchanged.

Activations, bias, decoded shared values, accumulation and outputs are F32. K must be divisible by 32; dimensions are 1..16384. Nonfinite F16 scales, invalid shapes, incorrect source lengths and cancelled contexts fail admission. The owned immutable allocation and kernel retain the established close/in-flight/drain contract. This standalone operator has no `VkF32Plan.Stage`, model integration, implicit fallback or default selection. Source bytes may be released after construction.

Packed GPU storage is 0.75 bytes/value versus 4 bytes/value for widened F32, a logical 81.25% reduction before driver padding. This is retained GPU weight storage only, not measured process RSS or whole-model allocation savings. Packing costs and model integration must be included in future whole-request comparisons.

## Numerical and latency evidence

Starting revision: `ae2c7ee56744c82b9020ee96467c9abfa72c253f`. Model: original Turbo Q5_0 `394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2`. Four exact tensor slices from encoder layers 0 and 31 FC1/FC2 were extracted after full-model pin verification. `model-tensors.json` records names, geometry, offsets, byte counts and slice hashes. Native tests admit a pinned tensor only for its matching K/N shape; remaining shapes use deterministic synthetic blocks.

The independent original-GGML fixture contains 64 blocks/2048 values. Offline repacking/reconstruction matches all original decoder values bitwise. Native identity projection covers the same fixture. Small/tail and full-size projections compare against widened original values with identical per-output K order; timed full outputs are bit-identical. Small scalar-reference tolerances remain `2e-5` absolute/relative. Bounds sentinels, cleanup, precancel/reuse and mocked mid-submit cancellation/drain/close tests pass. No trained transcription, VAD or word-timing arm runs through this operator.

The retained operator was exercised five times per pinned tensor. Each invocation used six warm ABBA/BAAB samples per arm and shape. Host dispatch/fence completion is timed; preparation, repacking, transfers, result validation and downloads are outside those kernel timings. Per-model pooled medians across those repeated samples gave about 3.8–6.4% less time for FC1-sized projections and 4.7–7.6% less for FC2-sized projections. Only the matching shape uses that model tensor; square synthetic projections were effectively tied. Exact samples, matching tensor geometry and ranges are in `timings.json` and native environment files.

Earlier variants are preserved: grouped unaligned source decoding was 6.7–12.9% slower; four-lane aligned loop decoding was 0.2–6.2% slower. Explicit four-lane expansion was only about 0.3–1.4% faster. Eight-lane/two-byte decoding produced the retained gain. This is grouped source decoding, not optional FP16 arithmetic or cooperative-matrix execution. The initial signed-conversion shader hit unsupported opcode 111; exact conversion of bounded unsigned values followed by subtraction avoided it. The opcode envelope was not widened.

## Structural changes tested after user feedback

### Larger command plans

An explicit temporary encoder combined consecutive layer stages while retaining their order, kernels, tensor lifetimes and barriers. A 64-stage batch passed tiny synthetic graph checks but exceeded the existing one-second submission wait budget on trained JFK. Native submission remained in flight; the process failed and its exit/cleanup errors are preserved. No wait budget was relaxed.

A 24-stage batch stayed within the budget and matched baseline text/tokens/segments/times for five JFK requests. Median request time was 7.081→7.045 s (−0.51%). That single process-pair gain is too small to accept. The mode, benchmark selection and graph tests were removed; their source patch is retained as rejected evidence. Defaults are unchanged.

### Packed projection plus GELU

A shader placed the existing erf-GELU directly after packed projection+bias, eliminating the intermediate F32 write/read and a dispatch. The first form accidentally imposed `precise` on projection arithmetic, changing contraction and failing exact parity on a small shape. The corrected form retains the prior projection semantics and passed all seven synthetic/independent shapes bitwise against the two-dispatch packed projection+GELU baseline.

The useful FC1 shape was effectively tied (+0.15% time); the other large shapes measured −1.27% and +4.31%. Fusion increased register/instruction pressure enough that removing memory traffic did not give a useful measured gain. These figures compare standalone dispatches; there was no trained fused encoder. The shader and diagnostic were removed from runtime selection and retained only as evidence.

## Shader gate correction

The full rebuild check discovered that archived score-ILP evidence had `pass:false` for `rope_sequence_f32`, despite the earlier all-pass statement. Stored and embedded modules had an extra `NonWritable` decoration on the read-only cosine/sine input variable. The member already had the correct read-only decoration. Regenerating from the unchanged GLSL removes that redundant marker; arithmetic, input/output bindings and instructions are unchanged apart from decoration ordering. Stored and embedded copies were regenerated together, shrinking 3344→3332 bytes.

The initial failure is retained; the prior validation text is corrected without rewriting old evidence. All **28** stored/embedded/rebuilt modules now validate and normalised-match with the recorded tools. Native RoPE exact-prefix and prepared-row tests pass five repetitions. A broader first invocation included the pre-existing released-layer test and failed because `model/nemotrondiarization/testdata/jfk_full_layer1_q.f32.gz` is absent. That missing-fixture gate is unqualified; it was not counted as passed. The focused native synthetic checks and regenerated-asset contract passed.

## Verification and isolation

The operator's model-free admission/packing/lifetime tests passed ten times. Scoped statement coverage reports constructor 85.7%, packing 90.9%, Close 83.3%, closeLocked 75.0%, Forward 94.9%; remaining fault/cancellation branches are not all covered. This does not meet the repository's 95% new-helper target, so fault coverage remains open. Independent source review timed out at 100 seconds and supplies no approval.

Verified retained-tree gates: affected tests/vet, `make model-layout-check host-build host-vet host-test docs-check`, six shader-checker tests, all 28 offline compiler/validator comparisons, whole-tree `go test -race -p=2 -count=1 -timeout=180s ./...`, and Linux ARM64/RISC-V whole-tree builds. Cross-builds are compilation evidence only. Native operator tests covered four matching pinned tensor slices × five invocations; the original 2048-value identity gate and small/tail checks accompany every invocation.

The authorised Intel container retained four-CPU quota, `taskset -c 0-7`, 8 GiB memory/memory-plus-swap cap, no network and a read-only root. Qwen-idle and host available memory ≥6 GiB guards stayed active. No production binary, profile, service, feature capability or limit changed. Final check and race containers exited 0 without OOM. At 06:32 UTC, native/build/candidate containers were drained, only pre-existing `wrdp` containers remained, no Go process was active, and Qwen slot 0 was idle. `@llama` received explicit release.

## Evidence and next work

[Evidence directory](../../benchmarks/speech-foundations/vulkan-q5-grouped-20261001/) includes rejected source/patches, grouped shader prototypes, pinned slice manifest/extraction helper, native raw samples/states, structural comparisons/failures, partial coverage, original and corrected shader verification, runner and environments. Large model tensor slices are not committed; regenerate them from the full pinned model with the retained extraction helper. Offline fixture provenance is in [testdata](../../backends/vulkan/testdata/Q5_GROUPED.md).

Next: integrate packed storage through a checked encoder weight provider and plan-compatible lifetime boundary; measure preparation, memory, whole-request latency and quality on the same original model. Test a more substantial structural change only when measured costs support it. The original-engine speed target, full Vulkan+flash+VAD comparison, independent acoustic/timing, long-form and resume acceptance remain open. No goal completion is established by this kernel-only result.
