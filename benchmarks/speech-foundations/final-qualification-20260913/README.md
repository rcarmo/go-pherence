# Final speech qualification audit — 13 September 2026

The audited baseline passes the supported Linux model-free release checks and the bounded physical-Iris lifecycle/parity runs. A separately qualified follow-up promotes the narrow pure-Go WAV/AAC-LC media path. The combined branch still does not satisfy the complete release plan: strict Community-1 intermediate gates, representative multilingual quality, speaker-stage latency, full combined-job performance and broad corpus/word-attribution tests remain open or failed.

## Revision and environment

- Revision under test: `623a1a8a59cffe384fde1450a3e00f5ba92a910d` (`feat/speech-simd-vulkan`).
- Final merged verification revision before this evidence-only commit: `e43dd6f` (audit `e6cf735`; P8 commits `a10ca83` and `5f1cf99`).
- Physical device: `Intel(R) Iris(R) Xe Graphics (RPL-P)` through `/usr/share/vulkan/icd.d/intel_icd.x86_64.json`.
- `GO_PHERENCE_DISABLE_NVIDIA=1`, `GOMAXPROCS=2`, `CGO_ENABLED=0`.
- Model assets: pinned Whisper Tiny `169d4a4`, Turbo `41f01f3`, and Community-1 `3533c8` conversions.
- Media during the baseline qualification: FFmpeg 8.1.2. The supplemental P8 qualification now pins pure-Go `github.com/rcarmo/go-264@v0.0.0-20260913172724-2db88745d0e5`; this includes the earlier `ProbeMetadata` repair and qualified common-window masked stereo PNS. The MINDS-14 rows were restored from the pinned `40ce77c…` dataset-viewer records and checked against committed hashes.
- The three speech/LLM user services stayed inactive. Every timed process reports zero swaps. No process remained after the native runs.
- The host exposes no thermal-zone or RAPL energy counters to this session. GPU frequency and host memory/swap were sampled at 250 ms; those samples are diagnostic, not energy evidence.

## Physical-Iris results

All eight bounded native commands exited zero.

| Gate | Result |
|---|---|
| Community trained process recovery | Child was killed after the first window; a fresh process completed; PASS in 14.36 s |
| Community trained full 30 s run ×3 | 13.383404 s, 13.282948 s, 13.329849 s; median **13.329849 s** |
| Community result identity | All three SHA-256 values `111e98ad0a6d1fb5dc61bcf6db4ff0d3d44485cc66c09efcabf7736b7d8f4ee3` |
| Community Vulkan ownership | 87,219,248 bytes / two allocations while live; exact cleanup in every test |
| Tiny trained encoder | Short boundary/decoder comparisons and three full-shape repeats PASS; repeat output SHA-256 `08125fb8…488` |
| Tiny JFK whole inference | Warm median 0.519284 s for 11 s: **21.183× realtime**; CPU/GPU output exact; WER 0/22; eight samples/path |
| Tiny public corpus | CPU/Vulkan tokens/timestamps exact for PT/PT/FR, silence and JFK compositions |
| Turbo promoted F32 register tile | 14.252234 s → 8.283896 s median, **1.720×**; bit-exact encoder and exact retained four-fixture output |

The Tiny JFK result exceeds the proposed 8× ASR threshold on one short English fixture. It is not the planned multilingual corpus acceptance result. Tiny quality is poor on the retained human MINDS rows: Portuguese WER is 80% and 77.78%; French WER is 40%. Default digital silence emits `you`; the explicit exact-zero skip removes it and preserves the 44-word three-window JFK composition.

The optimized Community hybrid is **2.25× realtime** on the 30 s public sample. It is 3.69× faster than the retained 49.235 s Go CPU median, but it does not meet the proposed ≤2.1 s speaker-stage target and is slower than the historical 3.10–3.19 s Rust/ONNX observation. That comparison is not balanced in one window.

## Supported release checks

The following commands pass on Linux/amd64:

- `speech-foundations-check`
- `speech-vulkan-offline-check`
- `speech-vulkan-community-check`
- `speech-vulkan-community-server-check`
- `speech-vulkan-encoder-check`
- `speech-sincnet-fma-check`
- `speech-community-gemm-check`
- `speech-job-check`
- `speech-job-http-check`
- `speech-job-cli-check`
- `speech-job-serve-check`
- `speech-media-integration`
- `speech-quality-freeze-check`
- `speech-community-corpus-contract-check`
- Community Linux/arm64 and Whisper Linux/arm64 test compilation

Whisper Windows test compilation still fails in pre-existing Unix-only Vulkan, NVIDIA and memory-advice packages. Windows is not a supported neural serving target in the current plan.

## Failed gates retained without tolerance changes

- Strict lowered SincNet: two wave/stride-10 cases fail at index 1717 with max absolute error `0.00038086623` against the fixed `0.0002` gate. Twelve related cases pass. Scalar and SIMD FMA remain bit-identical. A source and independent review classify the residual as unsupported backend-specific PyTorch 2.14/MKL blocked-tail arithmetic: reproducing it would require non-portable kernel emulation or a forbidden shape correction.
- Strict trained segmentation: `silence-1s` and `public-10-20s` fail intermediate gates. Endpoint hard-mask disagreement remains zero.
- Strict trained embedding: all four retained cases fail one or more frontend/trunk/support intermediate gates. Endpoint embedding checks used by the saved end-to-end comparison remain within their separate limits.
- Strict default tie policy rejects 84 ambiguous frames in the only trained diarization fixture. The complete result uses the explicit `LowestIndexTies` diagnostic policy.

These failures prevent calling Community-1 graph fidelity qualified, even though the saved final segmentation/mask/turn/DER result is stable.

## P0–P8 audit

| Phase | Status | Evidence / missing requirement |
|---|---|---|
| P0 reference freeze | Pass for implementation inputs | Hash-pinned models/configs, licences and provisional quality targets are checked. Final broad-corpus quality budgets were never ratified. |
| P1 exact CPU contracts | Partial | Exact Whisper 80/128-bin frontend and model formats pass. Community strict intermediate gates fail. |
| P2 execution foundations | Pass for supported Linux paths | Checked Plan 9 kernels, Vulkan ownership/barriers/arenas/plans, static shaders and native Iris tests pass. Race testing and wider architecture execution are incomplete. |
| P3 Whisper performance slice | Partial | Resident F32 encoder and CPU decoder are implemented. JFK reaches 21.18× realtime and Turbo encoder improves 1.720×, but representative multilingual quality and the planned matched ASR gate are not passed. Q8 remains research-only after natural long-form parity failures. |
| P4 Community-1 parity | Failed | Full graph, PLDA/VBx and turns exist; one saved DER result matches. Strict intermediates, default tie behaviour, broad annotated corpus, JER/word attribution and empty/noisy/overlap quality are not qualified. |
| P5 Community acceleration | Partial/failed target | Vulkan convolution/LSTM and Plan 9 SincNet are measured and preserve final output. The 13.33 s speaker run misses the ≤2.1 s historical target. |
| P6 integrated service | Partial | Durable queue, recovery, HTTP/CLI/UI, explicit Vulkan profiles and model-free lifecycle tests pass. The trained combined server path, general word alignment, long-file Community resume and host-wide resource coordination are not qualified. |
| P7 performance qualification | Failed/incomplete | No comparable ≥5× complete-pipeline result, no authorised private long-file run, no broad long/multilingual corpus, and no energy/thermal measurement. The only Community timing misses its target. |
| P8 pure-Go media replacement | Pass for declared subset | Provider timing API `ProbeMetadata` corrected the consumer's pre-trim/edited-extent misuse. Schema 2 requires an explicit backend; shipped profiles select the pinned pure-Go WAV/progressive AAC-LC provider, while FFmpeg remains explicit rollback. All six consumer WAV/AAC arms and durable resume/cancellation/profile-to-VTT gates pass. Provider CI run 34771655148 adds a second 8-case FFmpeg matrix, qualified common-window masked stereo PNS, and native ARM64 race coverage. Undeclared containers/codecs remain fail-closed. See `../go264-default-promotion-20260913/README.md`. |

## Release decision

`not qualified` for the complete plan. The implemented Linux paths are retained and the acceleration work is valid within its stated scope. Release completion requires at least:

1. fixing or formally revising the strict Community-1 intermediate contract with new independent evidence;
2. meeting or explicitly revising the Community speaker-stage target;
3. representative multilingual ASR quality and paired performance evidence;
4. broad annotated diarization, overlap and speaker-attribution evidence;
5. comparable integrated cold/warm, recovery, long-form, memory and energy/thermal trials.

`SHA256SUMS` covers every retained raw log, timing record, monitor trace and result document in this directory.

## Post-merge verification

The final merged speech branch was checked after the P8 merge with `CGO_ENABLED=0`, `GOMAXPROCS=2` and NVIDIA disabled:

- all 34 commands underlying the supported speech Makefile gates passed; `make` itself was unavailable in this host session, so the exact recipes were executed directly with the bundled Go toolchain;
- the final audit and P8 evidence manifests verify;
- the pinned public provider module downloads from the Go proxy with module sum `h1:+LFcHlJ5LZzlbjHfNMmsHjHvv3hTXl2bF2idL95/Nsg=`;
- Linux/ARM64 test binaries compiled for Community-1, Whisper, media and speech-job packages. The server command remains intentionally Linux/amd64-only and fails its cross-build because its profile references Linux/amd64-only trained runtime types;
- a repository-wide `go test ./...` remains red in unrelated SpaceMIT, DiffusionGemma and Qwen packages. Repeating exactly those packages at baseline `623a1a8` and at the merged revision produced the same twelve test failures; the speech merge did not introduce them;
- consumer race execution is not claimed under the required `CGO_ENABLED=0` setting because Go's race detector requires CGo. The provider's published amd64 Actions job supplies full race coverage for go-264.

The successful supported-gate log and explicit nonzero constraint/baseline comparison logs are retained here. None of these results changes the `not qualified` complete-plan decision above.
