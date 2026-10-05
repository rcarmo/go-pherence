# Pocket TTS bundle association — 30 September 2026

The Linux single-writer association candidate passes six tiny temporary-filesystem tests and 43 subtests. It publishes an association between opaque serialized model bytes and run-state metadata; it does not decode models, run training or prove power-loss durability.

Tested parent: `a90aec30567d485784f2e04d1803c543c9f2817f`, plus separate untracked `model/pockettts/training_bundle_linux.go` and `training_bundle_linux_test.go`. Both match the archived tested source. Direct source review accepted this bounded layer; commit/publication and integration require separate disposition. Accepted preflight/run-state and the eight staged inference paths are unchanged.

## Commit point and caller contract

`PublishTrainingBundle` requires an existing directory, one exclusive writer, a coherent immutable model stream and single-owner run state. Limits bound model bytes to an explicit ceiling of at most 16 GiB and sidecar bytes to at most 2 MiB; the pointer is at most 4 KiB. Disk availability, RSS and a blocking `io.Reader` deadline are not bounded by those ceilings. Supply trusted finite readers and stop/join updates before serialization.

The layer checks the existing 16-byte magic, version 1 and model step ≤MaxInt, then streams the full-file SHA. This validates association, not the binary trailer, tensors, optimiser settings or numerical state. The caller separately serializes/decodes/validates model state.

Exclusive no-follow generation creation never truncates old files. Publication order is:

1. Model write, sync and close.
2. Sidecar snapshot bound to actual model filename/hash/step, then write, sync and close.
3. Generation-directory sync.
4. Exclusive temporary pointer write, sync and close.
5. Atomic replacement of `current.json`, then final directory sync and close.

`current.json` is the commit point. Failures before rename retain the old pointer; partial new files are unselected orphans, never implicitly loaded or reused. Once rename succeeds, any subsequent error returns `TrainingBundlePublishedError`: the new coherent pair is visible but durability is uncertain. Do not automatically rollback, delete generations or retry after that outcome. Cleanup/retention is not implemented.

`OpenTrainingBundle` follows only the committed pointer, checks strict schema/names/byte counts/digests/steps and known link/special-file rejection, restores expected run-state declarations, then rewinds and returns the **same verified model descriptor**. The caller owns its close and must separately decode/validate the bounded checkpoint before mutating live state. The library cannot prevent hostile in-place writes; immutable single-writer generations are a precondition. It does not verify that declared platform/config/corpus identity was actually used during training.

## Bounded CPU gate

From repository root, this exact reviewed command ran once at 2026-09-30 23:39:04.555666811–23:39:06.423619323 UTC, exit 0, six named top-level passes and 43 subtest passes, no skips:

```sh
timeout -k 5s 120s env -i PATH="$PATH" HOME="$HOME" \
  GOENV=off GOFLAGS= GOTOOLCHAIN=local GOPROXY=off GOSUMDB=off \
  GOMAXPROCS=2 GO_PHERENCE_DISABLE_NVIDIA=1 GO_PHERENCE_DISABLE_VULKAN=1 \
  nice -n 10 go test ./model/pockettts \
  -run '^TestTrainingBundle(RoundTrip|CollisionAndAdmission|PrePublicationFailures|PublishedUncertainty|CorruptionAndBinding|SpecialFiles)$' \
  -count=1 -p=1 -parallel=1 -timeout=30s -v \
  -o /workspace/tmp/pocket-training-bundle-20260930/tests/
```

No asset/test opt-ins were inherited. The reviewed dependency/init closure is CPU-only. Temporary fixtures contain 65-byte opaque magic/version/step prefixes and tiny pure run-state metadata; they are explicitly **not decoder-valid trainable checkpoints**. All writes, symlinks, FIFOs, renames and removals are confined to `t.TempDir`.

Before/after hooks surround real write/sync/close/rename operations and verify resulting old/new file associations. They simulate boundary errors, not actual OS sync/close failures, process termination or power loss. Short-write detection is tested at the bounded-writer wrapper; failing Reader propagation is tested at the publisher. Root-close errors are source-reviewed but not hook-exercised. No general memory/filesystem/crash-safety proof follows from these cases.

Compile-only whole-tree vet/build and Linux arm64/riscv64 builds passed. No race/broad test suite, real publisher/checkpoint/model, trainer, codec, recordings, dataset or GPU run followed. Immutable source/log/meta/binary evidence is preserved in `/workspace/exports/pocket-training-bundle-20260930/`.

Actual data/latent preparation, full loss sampler, training runner, recorded-voice quality and Vulkan/PTX backward/optimiser paths remain separate work. The inference listening/publication and GPU/recovery holds remain in effect.
