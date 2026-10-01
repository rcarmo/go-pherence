# Streamed Q5 FFN preparation — 1 October 2026

Streamed packed FFN preparation removes 314.6 MB of cumulative Go allocations and reduces paired set-construction median by 7.19%. The encoder kernels, retained native weights and inference outputs are unchanged. CPU weight widening remains; the original-engine speed goal is still unmet.

## Construction change

Previously the explicit original-Q5 encoder collected all 64 FC1/FC2 raw matrices, then materialised all aligned packed copies before uploading to one final Vulkan allocation. `NewVkLinearQ5SetStream` admits shape/alignment/aggregate bounds and allocates the final native storage before invoking a synchronous reader once per matrix. Original source-value identity checking remains in the model callback. Each matrix is written directly into its final unpublished offset; raw arrays and callbacks are not retained by the owner.

The callback runs while the Vulkan lane is held and must not re-enter Vulkan. Reader errors, short extents, nonfinite scales, cancellation and panic trigger teardown of unpublished storage/kernel. Panic propagates after cleanup; there is no usable partial result. Cleanup failure preserves a returned owner with the error for retrying Close on normal error returns. The existing array-based set API remains available. Default model constructors and serving configuration are unchanged; only the already explicit original-Q5 FFN path switches preparation.

No F16 activation rounding, requantisation, stage reordering or inference shader change is introduced. Final native packed FFN storage remains 314,572,800 bytes; total encoder weights stay 1,184,890,880 bytes and scratch stays 93,696,000 bytes.

## Paired preparation and allocation evidence

Starting revision: `6e944fb1ec8e2f39c12b02e9d4ba40f015093d1e`. Original Turbo Q5_0 model pin: `394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2`. Go: `go1.26.2 linux/amd64`.

Five alternating stored/stream pairs used the same already-loaded widened CPU model and pinned open file, identical source-identity checks, kernel and final native storage. Each owner closed and returned native counters to baseline before the next arm. Timed construction includes raw matrix reads, identity checks, packing, pipeline/storage allocation and upload, but excludes CPU model loading and teardown. These are ten construction calls in one process, not ten independent cold launches.

| Median per set construction | Stored arrays | Streamed |
|---|---:|---:|
| Host time | 1.088650 s | 1.010374 s |
| Cumulative Go allocated bytes | 602,965,800 | 288,388,000 |
| Go allocations | 335 | 267 |
| Retained native packed bytes | 314,572,800 | 314,572,800 |

Time falls 7.19%; cumulative allocated bytes fall 314,577,800 (about 52.2%). Most of that saving is the eliminated complete aligned packed temporary. Raw reads still cumulatively allocate about 288 MB, but only one 4,505,600-byte matrix is directly live in the callback at a time instead of retaining all raw matrices. The maximum structural raw-plus-repacked temporary extent drops from about 602,931,200 to 4,505,600 bytes (about 598.4 MB removed). GC, driver padding, stale unreachable objects and CPU model storage can change actual peak RSS; this is a structural temporary bound, not a measured RSS saving. Monitored container samples are retained separately.

## Whole-arm and numerical checks

A separate five-request JFK pair against the retained array-preparation binary gave full encoder preparation `2.479 → 2.375 s`, full-arm time `42.320 → 42.272 s`, and request medians `6.920485 → 6.921276 s`. Request time is effectively unchanged. Full encoder preparation includes other weight uploads and graph setup; the isolated set comparison supplies the repeated attribution measurement. No inference gain or cold single-request acceptance is established.

All five JFK returned window structures match the retained binary exactly. Native trained synthetic-mel qualification matched all 1,920,000 hidden-state values against widened F32 FFN and passed early/middle/late cancellation, drain where required, reuse and native cleanup around 1,303 checked boundaries. JFK and two-group native Silero/gap-preserving word-timing runs each repeated five times and matched retained original-Q5 text, tokens, segment/word times, retained coordinates and eligibility exactly. These are regression checks, not independent acoustic/timing annotation or long-form accuracy.

## Failure checks and validation

Offline tests cover shape/count/alignment and aggregate budget, direct packing vs independent retained packer, source ownership, canaries, invalid source/destination extent, nonfinite scales, early/block/final cancellation, callback errors, wrong lengths, reader cancellation and panic, rollback and no reader invocation before admission. Deterministic stream gates passed ten repetitions. Scoped coverage: packing 100%, geometry 91.7%, constructor 86.8%. Some native teardown/cleanup-failure and host-only branches remain outside the 95% helper target; fault coverage remains open. A focused source judge timed out after 60 seconds and supplies no independent approval.

Verified retained-tree gates: `make model-layout-check host-build host-vet host-test docs-check`, affected tests/vet, whole-tree `go test -race -p=2 -count=1 -timeout=180s ./...`, and whole-tree Linux ARM64/RISC-V builds. Cross-builds are compilation evidence only. Inference SPIR-V is unchanged from the previously verified 28-module gate.

The coordinated Intel runner retained four-CPU quota, `taskset -c 0-7`, 8 GiB memory/memory-plus-swap cap, `GOMEMLIMIT=4GiB`, no network, read-only root, Qwen-idle and host available memory ≥6 GiB guards. No defaults, services or resource limits changed. The final gated container exited 0 without OOM or guard abort. At 07:44 UTC, candidate/native/build work was drained, only pre-existing `wrdp` containers remained, no Go process was active and Qwen slot 0 was idle. `@llama` received explicit release.

## Evidence and next work

[Evidence directory](../../benchmarks/speech-foundations/whisper-q5-stream-preparation-20261001/) contains preparation samples, allocation counts, same-work request reports, VAD/word regression references, native recovery, resource monitors, complete check logs, coverage, pinned commands/environment and hashes. `comparison.json` separates paired set construction, whole-arm setup and request boundaries.

`TestVulkanQ5PreparationComparison` requires `GO_PHERENCE_TEST_Q5_PREPARATION=1`, pinned original-Q5 and HF metadata paths, named Intel device and timeout no longer than 300 seconds. Run only in a coordinated guarded window; ordinary tests skip native work. Reproduce with new report paths and do not overwrite retained evidence.

Removing CPU FFN widening needs a separate checked model/source contract; supplying fabricated F32 slices to bypass layout admission is not acceptable. That work, independent quality/word timing, long-form/resume and the original Vulkan+flash+VAD speed comparison remain open.
