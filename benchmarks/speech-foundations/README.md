# Speech foundation verification

CPU-only implementation checkpoint on Go 1.26.2, linux/amd64. No model inference, GPU initialisation, service restart or throughput benchmark.

- `verification.json`: 54 passing test/subtest events, no failures, one opt-in skip subsequently run successfully via the media target. Includes affected-package vet/build scope and limits.
- `focused-check.txt`: `make speech-foundations-check speech-media-integration` output. Tiny synthetic media only.
- `full-build-known-errors.txt`: full-tree compile failures, reproduced identically on an untouched archive of `d08ce322292847a7978dcace1cc73a50d0bf8450`.
- `stream-verification.json`, `stream-focused-check.txt`: subsequent bounded PCM-reader/window checkpoint. 67 passing test/subtest events, zero failures, one opt-in skip subsequently exercised successfully. This is a separate test selection, not a sum with the earlier count. Includes reader/window allocation checks and four-hour coverage geometry without inference.

Run with `GOMAXPROCS=2 CGO_ENABLED=0`, Go `-p=1`, a writable `TMPDIR`/`GOTMPDIR` and independent caches. FFmpeg tests use their own temporary directory and quarter-second waveforms, never private files or live service endpoints. The NumPy oracle generator used the pinned Transformers numerical source and synthetic PCM; no model import.

The exact frontend reuses existing checked Plan 9 SIMD `Ddot`. No new SIMD kernel, Vulkan backend, Community-1 port or model-performance result is established by these tests. Legacy inference defaults are retained until real-checkpoint validation.
