# Pocket TTS explicit training inputs — 1 October 2026

The bounded explicit-input preparation helper passed six CPU tests and 135 subtests, with no skips or failures. It validates and copies supplied values into caller-owned sample buffers. Stochastic sampling and executed training readiness remain unverified.

Tested parent: `a90aec30567d485784f2e04d1803c543c9f2817f`, plus separate untracked `model/pockettts/training_samples.go` and `training_samples_test.go`. Both match the tested source archive. Independent direct review accepted this narrow helper; integration and publication require separate decisions.

## Buffer contract

`PrepareTrainingSamplesInto` requires positive `MaxFrames` and `MaxNoiseElements`, positive channels and `0 < validFrames <= frames <= MaxFrames`. It checks `frames*channels`, its float32 byte range and the noise-element ceiling before scans or writes. Positive channels and the valid-prefix bound make active/time views no larger than that checked range. These ceilings bound helper work, not trainer resident memory or RSS.

Destination views have exact full-frame lengths: mask and each time array contain `frames` elements; noise contains `frames*channels`. Input draws contain only active rows: noise has `validFrames*channels` elements and each time array has `validFrames`. Larger caller arenas must supply bounded views.

Every shape, float alias and value check precedes mutation. Writable float destinations may overlap neither each other nor any input. Read-only inputs may overlap each other; adjacent disjoint views are admitted. Noise must be finite, diagonal time must be in `[0,1]`, and pairs must satisfy `0 <= s <= t <= 1`, including equal endpoints. Rejected calls leave destinations and sources unchanged.

Success copies active float32 values exactly, including signed zero; creates a contiguous valid-prefix mask; and writes positive zero only to sample noise/time padding. Storage outside the supplied views and sources is untouched. The helper retains no references, allocates no destination state and consumes no RNG or run-state proposal. It appends no EOS row and transforms no latent/target, noise scale, objective or gradient. Existing physical/valid lengths determine the EOS boundary elsewhere.

Callers must supply normal typed bool/float storage without unsafe-forged cross-type aliases and must not mutate buffers concurrently. Such forged aliases and concurrent mutation are unsupported.

## Pinned input evidence

The 84,996-byte `testdata/training_step_pytorch.json` is read with a 128 KiB ceiling and checked for schema 1, shapes, recorded generator and upstream revision. Its independent generator pins `kyutai-labs/pocket-tts@0acce6b2f390150267557770d2098c5caa9a18ac`, records PyTorch `2.13.0+cu130` and overrides `sample_t`/`sample_s_t` with explicit times. The fixture has three physical frames, two valid rows, mask `[true,true,false]` and zero sample padding. Tests consume only input fields; no model, loss, gradient or oracle generation runs.

Generator and fixture last changed in local commit `961c645958cdaeb690cfbbef74186b80698d6dd5`. Input provenance SHA-256:

- Fixture: `2774178aba94bfcfea47932d557c53c5aaf2a43ba32a12fbe1d7a463070d22ed`.
- Generator: `9b0aea68f07ac057c4bc447c725030f74d60011caf7d719b19c1049f9b52e16b`.

These hashes identify fixture/source inputs. Exact comparisons enforce the copy/signed-zero/padding and discrete metadata contracts; they do not qualify generated speech.

## Single bounded gate

This command ran once from repository root at **2026-10-01 00:09:11.101850695–00:09:11.344265852 UTC**, exit 0:

```sh
timeout -k 5s 120s env -i PATH="$PATH" HOME="$HOME" \
  GOENV=off GOFLAGS= GOTOOLCHAIN=local GOPROXY=off GOSUMDB=off \
  GOMAXPROCS=2 GO_PHERENCE_DISABLE_NVIDIA=1 GO_PHERENCE_DISABLE_VULKAN=1 \
  nice -n 10 go test ./model/pockettts \
  -run '^TestTrainingSamples(ExplicitFixture|ShapeAdmission|Values|Aliasing|PaddingOwnership|NoDraws)$' \
  -count=1 -p=1 -parallel=1 -timeout=30s -v \
  -o /workspace/tmp/pocket-training-samples-20260930/tests/
```

| Test | Subtests | Scope |
|---|---:|---|
| `TestTrainingSamplesExplicitFixture` | 0 | Pinned explicit inputs and full-frame layout |
| `TestTrainingSamplesShapeAdmission` | 42 | Geometry, ceilings, overflow and exact lengths |
| `TestTrainingSamplesValues` | 20 | Finite/range admission, equal endpoints and signed zero |
| `TestTrainingSamplesAliasing` | 69 | Exact/partial writable aliases, read-only aliases and disjoint views |
| `TestTrainingSamplesPaddingOwnership` | 4 | Prefix/padding, guards, source preservation and result ownership |
| `TestTrainingSamplesNoDraws` | 0 | Discrete proposal/abort/restore state and helper allocation average |

All six top-level tests and 135 subtests passed without skips. Fresh environment controls excluded asset/test opt-ins. The reviewed host dependency/init closure contains CPU capability detection and static YAML tables, with no NVIDIA/Vulkan loader. Metadata cases use tiny PCG/JSON state and dummy bindings; they invoke no checkpoint publisher or trainer.

`AllocsPerRun(100)` returned zero for the warmed helper with pre-owned three-frame/two-channel buffers. Go 1.26.3 performs one warm-up, forces `GOMAXPROCS=1` during measurement and integer-divides total mallocs by 100. A zero result can therefore hide fewer than 100 aggregate mallocs. This is a narrow integer-average check; it supplies no universal zero-allocation, whole-training memory or historical inference-outlier attribution.

Focused vet and host/Linux arm64/riscv64 test-binary compilation passed separately; foreign binaries were not executed. Command, revision, fixture provenance, UTC, log, tested sources, binaries and dependency/compile evidence are archived in `/workspace/exports/pocket-training-samples-20260930/`. No additional tests ran.

## Remaining training work

The pinned upstream stochastic sampler/model/dataloader sources have not been located locally. The fixture's deterministic overrides cannot establish Gaussian scaling, timestep distribution, pair coupling, selection rules, random draw consumption or PyTorch RNG-sequence equivalence. No production random adapter is implemented, and the accepted run-state schema is unchanged.

Runner integration, real local voice training, recorded-voice quality and Vulkan/PTX training remain unfinished. This slice ran no downloads, model/data/latent preparation, oracle, trainer or native backend. Accepted candidates, index and prior evidence remain untouched; commit/publication, listening and GPU/recovery holds remain in effect.
