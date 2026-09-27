# Needle 3 direct CQ on native ARM64

The opt-in direct CQ loader passed bounded native ARM64 checks on a CIX P1 CD8160 on 27 September 2026. It cut observed memory use against the decoded and hybrid loaders, but did not improve twelve-token cached throughput. RISC-V execution was not available; Linux/RISC-V cross-compilation alone does not qualify RVV.

## Transfer and parity

Linux/ARM64 test binaries were cross-built from `3713310c` with Go 1.26.2 and `CGO_ENABLED=0`. The CIX P1 ran `GOMAXPROCS=2 nice -n 10` with GPU execution disabled. Transferred binaries, fixture data and the 35,335,380-byte released archive were SHA-256 checked before execution. The archive hash was `c9d915eca282ed42d1a09b143b592adb4cc6744ffe2d294adf5cfc5548170c38`. Binary hashes were `892b33e7746b2499c276db485028c3f6ee025320f0ae422fe43649cf2c330b6a` (model), `25b3dfa13fc9805e47fc9b99f00c39adc7c709f87f6ed5e02b592a25d34c1062` (loader), `80da926e5bacf20a6c6beb6850efe4444948967876f49b06a8bc67d98904aa3f` (SIMD) and `4aac80ec63bc2c0de395543b56ee42bd578dc50c9d2065e36e8669a6e2d67a69` (dot comparison). The test bundle also contained the repository's Needle fixtures; the first run from the wrong working directory failed to find relative fixture paths and was rerun from each package directory.

Three native repetitions passed selected tiny-archive packed/compact/coverage tests, loader CQ validation and SIMD CQ tests. The released archive passed the twelve-token full/cached comparison and the previously recorded chat-prompt continuation against the decoded path. Its bit inventory is 115 CQ2 and seven CQ4 records. The released archive has no auxiliary heads; head parity is limited to the tiny fixture. The active ARM64 SIMD capabilities reported NEON, dot, SGEMM and pack support. A separate CQ2/CQ4 test compared native-dot matrix products with scalar-dot products on 768-column shapes; three repetitions passed under the test's `5e-4 + 5e-4*abs(reference)` per-element bound. Unpacking and Walsh transforms remain scalar in this path. The native binaries were not race-instrumented; the host CPU race suite passed separately.

## Fresh-process measurements

Each mode loaded the same pinned archive in three separate processes, in rotated order. Samples include five warm two-token cached calls after loading and `debug.FreeOSMemory`. VmHWM and VmRSS came from `/proc/self/status`, not an external process sampler. These values include runtime and loader allocations and are specific to this host and workload.

| Mode | Median VmHWM (KiB) | Median post-load VmRSS (KiB) | Median post-warm VmRSS (KiB) | Median load (ms) | Median warm two-token call (µs; 15 calls) |
|---|---:|---:|---:|---:|---:|
| Decoded | 840,740 | 496,008 | 497,072 | 1,734 | 63,533 |
| Hybrid | 840,672 | 495,880 | 496,888 | 1,603 | 73,833 |
| Direct | 382,044 | 308,020 | 308,684 | 1,825 | 73,878 |

Direct minus hybrid median peak was −458,628 KiB (54.6% of hybrid peak). Three separate fresh-process benchmark samples per mode ran five twelve-token cached sessions each, after model loading and decoder construction. The median times were 357,224,134 ns/session decoded, 422,149,719 hybrid and 420,323,487 direct. Direct packed was about 17.7% slower than decoded in this bounded run. Per-session allocations were 2,248,784 bytes/11,620 decoded, 6,338,742 bytes/12,951 hybrid and 6,337,104 bytes/12,951 direct. The board had other workloads; `nice -n 10` and the rotated order reduce, but do not remove, scheduling effects. No sustained multi-hour or concurrency throughput was measured.

An additional released-model cancellation check shared one direct-loaded model across four independent cached sessions. Each ingested a token, cancelled after provisional layer work on the second token, verified no cache commit, retried, reset and compared with the decoded-reference logits. One Intel ordinary and one Intel race run passed; three native ARM64 ordinary repetitions passed from SHA-checked binary `e95626c857fef153dff8b1f84f671b71b3e7400175cdcd35bcec4c5d2078bd06`. It does not measure hard cancellation latency, days-long retention or a sustained concurrent load.

Raw logs and SHA manifests remain on the board under `/home/agent/needle3-native-20260927/`. The Intel [direct archive record](needle3-packed-archive-parser-20260927.md) uses a different host and `GOMAXPROCS=6`; its timings should not be combined with these ARM64 figures. Native RISC-V/RVV, vectorised CQ unpack/Walsh, long-running concurrent cancellation and broader prompt quality need separate validation before production admission.
