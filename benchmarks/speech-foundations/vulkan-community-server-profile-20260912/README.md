# Explicit Community-1 Vulkan server profile — 12 September 2026

This checkpoint adds an opt-in experimental Community-1 hybrid Vulkan profile to `speechjobserve`. It is model-free and device-free: no Vulkan initialisation, trained model, audio, corpus, service, deployment, push, pin, default, or numerical gate changed during verification.

## Implemented

- `community1.NewVulkanDiarization` composes fixed-window CPU SincNet/head, resident Vulkan LSTM and ResNet trunk, CPU mask-dependent pooling/projection, and existing PLDA/postprocessing.
- Construction validates graph/window geometry before native allocation, constructs segmentation before embedding, rolls back in reverse order, and retains any failed native child for retry.
- `speechjob.NewVulkanCommunity1Stage` serialises whole-stage access, uses a fresh context to drain after every inference return, and keeps admission/resources held until idle is proven.
- After successful resident construction, the server explicitly clears its exclusively owned source segmentation and embedding neural tensors; CPU SincNet lowered filters and PLDA copies required by the hybrid remain resident. The release API is idempotent and does not claim immediate RSS reduction.
- Panic, device-loss, uncertain submission, and fatal drain errors quarantine the process rather than allowing CPU fallback or unsafe teardown. Eight actual child-process kill/reopen cases verify durable whole-result retry after process death.
- Close stops new admission, waits for active work, closes the composite model in reverse order, and retains failed resources for retry.
- Nested `profile.community.vulkan` configuration requires explicit enablement and experimental consent, a resolved device substring, backend SHA-256, and bounded drain interval.
- Simultaneous Whisper and Community-1 Vulkan profiles are rejected because the current backend has one process-global serialized device lane and no independent device-owner lifecycle.
- Stage identity includes the original Community model/payload/runtime identity plus the selected backend hash, resolved device name, and resident graph statistics.
- Metadata-only `--check` verifies assets and configuration without initializing Vulkan or constructing models.
- Existing CPU Community-1 and CPU Whisper behavior is unchanged when the nested Vulkan object is absent.

## Verification

Using `GOMAXPROCS=2` and the checked workspace Go toolchain:

- `go test ./models/speaker/community1 ./runtime/speechjob ./cmd/audio/speechjobserve` passed.
- Ten shuffled repetitions of those three packages passed.
- Ten shuffled repetitions of the eight process-kill recovery modes and focused runtime lifecycle tests passed; thirty shuffled focused server construction/configuration repetitions passed.
- Focused mock-only Community Vulkan and Vulkan operator checks passed.
- The six static shader-checker tests passed.
- A full `speechjobserve` package run passed in 5.755 seconds.
- `go vet` passed for the three affected packages.
- `git diff --check` passed.

The first unbounded full server test invocation inherited a host `GOMAXPROCS` value outside the test fixture's declared two-slot resource budget, causing the known resource-config failure and leaving a listener-wait test blocked until its timeout. Re-running under the target's documented `GOMAXPROCS=2` environment passed both focused failures and the full package. This was configuration, not model/device execution.

## Open gates

Native synthetic parity still requires the explicit `speech-vulkan-community-native-check` admission window. Trained checkpoint parity, strict SincNet/intermediate failures, device-loss process recovery, corpus DER/JER/SA-WER, long-file behavior, and CPU/full-Vulkan/hybrid whole-job performance remain open. This profile is not production qualified and was not deployed.
