# Pocket TTS run-state library — 30 September 2026

The pure CPU run-state candidate passes six bounded discrete-state tests. It preserves declared completed-step cursor/raw-RNG state; it does not update models, publish checkpoint bundles or establish executed training reproducibility.

Tested parent: `a90aec30567d485784f2e04d1803c543c9f2817f`, plus `model/pockettts/training_run_state.go` and `training_run_state_test.go`. Those two files match the archived tested source and remain untracked/uncommitted. Direct source review by `@default` accepted the library within this scope. The eight staged inference paths and six separate preflight paths are unchanged.

## API boundary

`NewTrainingRunState` starts at optimiser step zero. V1 accepts one row per completed optimiser+EMA update, constant LR, dropout disabled and no pending accumulation. Epoch-end normal form is `cursor == dataset_rows`; only the next proposal rolls over and shuffles. Snapshot/restore consume no draws.

Usage sequence:

1. Call `PrepareRow` to obtain an owned proposal; read its row and independent `TimeUint64`/`NoiseUint64` draws. These are raw PCG values, not a complete upstream loss/time/noise sampler.
2. Perform the model update through a future caller-owned runner.
3. Only after the optimiser and EMA complete successfully, call `CommitRow(proposal, completedStep)`.
4. Bind `Snapshot` to the actual supplied checkpoint generation filename, digest and matching step. Verify expected bindings before exposing a state through `RestoreTrainingRunState`.

Proposals deep-copy the permutation/RNG state. Abandoning one leaves its parent unchanged; commit rejects wrong-parent, stale/replayed proposals and step jumps. A failed or uncertain model update requires discarding the proposal **and** recovering a coherent prior model checkpoint. This library cannot roll back partially changed parameters, Adam moments or EMA.

Row/raw-draw accessors reject committed proposals, but do not reject every stale proposal. Their results are not authorisation to start new model work. The future runner must enforce freshness at its model-work boundary.

## Identity and snapshot limits

Caller-supplied identity includes source revision, Go version, GOOS/GOARCH/SIMD mode, SIMD backend and corpus-bundle/initial-model/tokenizer/Mimi/config/optimiser-EMA/options digests. Restore validates exact declarations; it does not independently verify runtime platform, on-disk contents or model settings. A corpus digest must bind audio/latent provenance and preprocessing/split policy, not only manifest text.

Mutable cursor state carries no stale checkpoint digest. Snapshot binds a copy to a confined relative generation filename and matching completed step. Two independent sidecar/model writes are not crash-consistent publication; immutable generations plus an atomic manifest/pointer remain future work. No files are opened by this library.

The JSON ceiling is 2 MiB, with at most 100,000 rows. Schema/algorithm/boundary IDs, null/duplicate/unknown fields, unsigned integer syntax/range, strings/digests, fixed PCG state, permutation uniqueness/range and step/epoch/cursor invariants are checked before exposing restored state. No runner, scheduler framework, rollback, gradient accumulation or backend integration is implemented.

## Bounded CPU evidence

From repository root, the reviewed exact command ran once at 2026-09-30 23:23:14.653016758–23:23:16.077549499 UTC, exit 0, six named passes and no skips:

```sh
timeout -k 5s 120s env -i PATH="$PATH" HOME="$HOME" \
  GOENV=off GOFLAGS= GOTOOLCHAIN=local GOPROXY=off GOSUMDB=off \
  GOMAXPROCS=2 GO_PHERENCE_DISABLE_NVIDIA=1 GO_PHERENCE_DISABLE_VULKAN=1 \
  nice -n 10 go test ./model/pockettts \
  -run '^TestTrainingRun(IndependentStreams|ResumeSequence|TransactionBoundary|BindingIdentity|StrictDecode|BoundsAndStepZero)$' \
  -count=1 -p=1 -parallel=1 -timeout=30s -v \
  -o /workspace/tmp/pocket-training-run-state-20260930/tests/
```

No asset/test opt-ins were inherited. The reviewed dependency/init closure is CPU-only; tests use tiny integer/PCG/JSON state, open no checkpoint/recording/model files and call no trainer/backend. Raw streams are compared with separately constructed standard-library PCGs; variable-draw resume tests cover cursor and epoch transitions. Exact comparisons apply to this discrete metadata/RNG contract, not generated speech. Fixed-seed shuffle self-comparison does not independently pin a golden permutation or prove all rejection-shuffle branches.

Compile-only whole-tree vet/build and Linux arm64/riscv64 builds passed. No race, broad test suite, training, codec, model, dataset, process-crash or GPU test followed. Immutable source/log/meta/binary evidence is preserved in `/workspace/exports/pocket-training-run-state-20260930/`.

Commit/publication and the inference listening disposition remain held. Numerical training reproducibility, durable bundle publication, local new-voice quality and Vulkan/PTX backward/optimiser support require separate work. GPU/recovery restrictions remain in effect.
