# Checked original-Q5 packed source — 1 October 2026

The checked packed-source path halves measured model-loading plus encoder preparation time, saving about 3.8 seconds and 5.35 GB of cumulative setup allocations. Inference kernels are unchanged; measured warm request medians are slightly slower. The original-engine speed target remains unmet. No default or production service changed.

## Explicit resident-only model contract

`LoadOriginalQ5ResidentChecked` accepts a caller-owned pinned open original GGML file and checked Whisper config. It validates the complete metadata inventory through the existing checked loader before reading F32 tensors. A private loading mode skips exactly encoder FC1/FC2 weight materialisation; biases, other encoder weights and the CPU decoder still load/validate normally. Original Q5 dtype, dimensions, row-block alignment and byte extents for every skipped matrix are independently checked.

The temporary encoder skeleton contains nil FFN weight slices, not fabricated zero-valued F32 arrays. A private packed-only Vulkan layout permits those declared packed matrix shapes only in original-Q5 mode; the normal layout continues rejecting missing weights. The same pinned file supplies raw blocks, which the stream constructor checks for finite scales and copies into final owned native storage. No checkpoint mixing or re-quantisation occurs; non-FFN data and packed FFN originate from the same file.

On success, the returned `Whisper.Encoder` is nil. CPU transcription rejects it; callers must supply the separately returned resident encoder. Host decoder flags and resident geometry are checked before return. File use is synchronous and caller-owned; the completed decoder/encoder retain no file handle. Errors return no model, and native teardown failure returns the partial encoder with an error for retrying Close. Normal `LoadModelSourceChecked` and CPU/default paths still require every weight.

The benchmark admits this through explicit `vulkan-original-q5-source`, legacy-value mode and retained original model pins. Model-load and preparation allocation counters are additive report fields. Bench setup uses the same private construction steps so those phases can be measured separately; a native trained test also exercises the public atomic loader.

## Matched setup and full-arm measurements

Starting revision: `19869ca76093ee399d214b3c1dd84714be339e38`. Go: `go1.26.2 linux/amd64`. Original Turbo Q5_0 model pin: `394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2`. Both arms use the existing packed GPU FFN encoder, score-ILP flash-style key32 online softmax, original stored values, F32 activation/decoder arithmetic, four CPU threads and `GOMEMLIMIT=4GiB`. Baseline materialises CPU FFN arrays and checks their identity during native preparation; source mode never materialises them.

Each arm is a fresh isolated process with five retained requests. Full-arm includes loading, preparation, requests and cleanup. Input/model pinning and matched decoder/VAD/word policies are recorded.

| Fixture/options | Baseline setup (s) | Source setup (s) | Baseline five-request arm (s) | Source arm (s) |
|---|---:|---:|---:|---:|
| JFK, no VAD/words | 7.650 | 3.814 | 42.140 | 38.456 |
| Portuguese, no VAD/words | 7.595 | 3.801 | 42.014 | 38.398 |
| French, no VAD/words | 7.597 | 3.801 | 39.308 | 35.685 |
| JFK, native VAD/words/gap preservation | 7.596 | 3.801 | 48.072 | 44.514 |
| Two groups, native VAD/words/gap preservation | 7.604 | 3.810 | 89.001 | 85.678 |

Setup falls about 50%; full-arm reductions are about 3.3–3.7 seconds. The initial exploratory pair is also retained. These are one process-pair per fixture, not five independent cold startups; request repeats establish output stability, not a cold-start distribution.

All five candidate and baseline requests per fixture return identical text, tokens, segment/window times and, when enabled, VAD retained coordinates, eligibility and every word time. Warm request medians rise about 0.23–0.62% across these pairs (for example JFK 6.883→6.915 s and grouped words 16.264→16.357 s). This is a setup/memory optimisation with no inference-speed acceptance.

## Memory boundaries

Logical retained CPU FFN arrays removed: `32 × 2 × 1280 × 5120 × 4 = 1,677,721,600` bytes. The loader also avoids transient decoded tensors/copies and repeated identity decoding. Cumulative Go allocations during measured load+preparation fall from about 10.64 GB to 5.29 GB (about 5.35 GB removed). These cumulative counters are distinct from retained bytes and RSS.

Native encoder stats are identical: weight storage 1,184,890,880 bytes, scratch 93,696,000 bytes, 34 plans and 390 stages. The rest of the CPU encoder is discarded after resident construction; this was also permitted by the prior resident PCM API. The observed maximum sampled JFK container memory was 5.507→4.163 GB, but periodic container samples are not true peak process RSS. Raw monitor logs and arithmetic are retained; no high-water precision is implied.

## Admission, native parity and checks

Offline tests prove that skipped FFN metadata still rejects shape, dtype, missing tensors and overlap before any value reads. Loaded FFN slices stay nil; normal CPU validation and normal Vulkan layout reject them. Decoder-only resident validation passes. Tests cover name placement, wrong Q5 metadata/type/extent, malformed config, cancellation and nil source/context. Selected gates passed ten times. Full checked-loader regressions also pass.

The public resident loader produced bit-identical 1,920,000 hidden-state values against the widened baseline; CPU fallback was rejected after construction. Decoder FC1 arrays compared bitwise. Early, middle and late cancellation/reuse across about 1,304 checkpoints passed drain where required, exact output and native cleanup. This trained test ran in an authorised isolated window, with file closure before inference.

Scope coverage for selected packed tests: source-name selection 100%, metadata-array checks 100%, metadata wrapper 40%, public loader 27.3%, shared checked-load helper 73.7%, packed layout helper 84.8%. Native success tests supply additional behaviour evidence but are not counted in that offline coverage. Native construction/cleanup fault injection and the 95% new-helper target remain open. A focused read-only judge timed out at 50 seconds; no independent approval is established.

Verified retained-tree gates: affected tests/vet, `make model-layout-check host-build host-vet host-test docs-check`, whole-tree `go test -race -p=2 -count=1 -timeout=180s ./...`, and whole-tree Linux ARM64/RISC-V builds. Cross-builds are compilation only. No inference shader module changes; the preceding 28-module verification applies to unchanged assets.

The Intel runner retained four-CPU quota, `taskset -c 0-7`, 8 GiB memory/memory-plus-swap cap, no network and read-only root. Qwen-idle and host available memory ≥6 GiB guards stayed active. No resource cap, default, model routing or service changed. Final gated container exited 0 without OOM/guard abort. At 08:19 UTC, native/build/candidate work drained, only pre-existing `wrdp` containers remained, no Go process was active and Qwen slot 0 was idle. `@llama` received explicit release.

## Evidence and remaining objective

[Evidence directory](../../benchmarks/speech-foundations/whisper-q5-packed-source-20261001/) contains initial/final pinned reports, semantic comparisons, allocation counters, memory samples, public-loader recovery, complete check logs, source tests/coverage, runner, environments and hashes. `comparison.json` keeps setup/full-arm/request and memory boundaries separate. Use fresh report paths when reproducing; existing evidence is protected from overwrite.

Native public-loader test: `GO_PHERENCE_TEST_PACKED_SOURCE=1`, pinned GGML/HF assets, named Intel device and `TestVulkanOriginalQ5SourceNative` with timeout no longer than 300 seconds. Benchmarks use `TestWhisperPerformanceGoalArm` with explicit source backend and the archived guarded environments. Ordinary tests skip trained native execution.

Matched original-engine speed, independent acoustic/word timing, long-form/resume and complete Vulkan+flash+VAD acceptance remain open. The warm-request deficit cannot be closed by describing setup savings as inference acceleration. Next speed work needs a measured algorithmic or compute-path change and a comparison with equivalent precision and timing boundaries.
