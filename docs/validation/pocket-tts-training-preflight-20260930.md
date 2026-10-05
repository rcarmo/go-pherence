# Pocket TTS training preflight — 30 September 2026

The Linux `cmd/audio/pockettrain` candidate passes a bounded synthetic CPU admission gate. It is read-only preflight source, with no released-model, recorded-data or executed training-readiness acceptance.

Parent: `a90aec30567d485784f2e04d1803c543c9f2817f`. Four source/test files remain unstaged, separate from the eight staged BF16 inference paths. Direct review by `@default` accepted this preflight-only source after the hardened gate. Commit/publication and the inference listening disposition are separate holds.

## Checked inputs

The [command README](../../cmd/audio/pockettrain/README.md) defines its syntax and size/path contracts. Unsupported Vulkan/PTX training or non-preflight mode is rejected before file access. No model loader, mmap, tensor materialisation, codec call, optimiser or backend initialisation exists in the command.

Model headers use bounded strict JSON types/keys, positive checked shape geometry, exactly two offsets and complete contiguous payload intervals; metadata is an object of unique string fields. File bodies are streamed for required SHA-256 provenance. Production also checks the Pocket six-layer student inventory. Hashes pin input assets; generated audio is not part of this gate.

Recordings are confined relative paths under `os.Root`, with known special-file rejection and descriptor-level nonblocking regular-file checks. The initial WAV contract is canonical 44-byte mono 24 kHz PCM16/F32. Streaming validation checks declared extent, finite/range/full-scale samples and explicit sample ceilings; it performs no resampling, downmixing, trimming or writes.

Aligned cuts use nearest-sample timestamp conversion and `ceil(samples/1920)` codec geometry. A 1.8-second target is 23 frames. A full five-second prompt is 63 frames and exceeds the existing 62-frame envelope; it is rejected without implicit trimming. Target is bounded at 375 frames. The output separates exact admitted sample/frame/token facts from estimated resident training bytes, which exclude checkpoint peaks and free-disk checks. `training_command_implemented` and `model_tensor_values_validated` are false.

## Synthetic execution

The initial exact ten-name gate ran 2026-09-30 22:59:40.434452315–22:59:41.404827547 UTC, exit 0, ten named passes and no skips. Its snapshot is preserved separately.

After reviewed parser/fixture hardening, this exact twelve-name gate ran once at 23:06:50.010913258–23:06:50.374530952 UTC, exit 0, twelve named passes and no skips:

```sh
timeout -k 5s 120s env -i PATH="$PATH" HOME="$HOME" \
  GOENV=off GOFLAGS= GOTOOLCHAIN=local GOPROXY=off GOSUMDB=off \
  GOMAXPROCS=2 GO_PHERENCE_DISABLE_NVIDIA=1 GO_PHERENCE_DISABLE_VULKAN=1 \
  nice -n 10 go test ./cmd/audio/pockettrain \
  -run '^TestPocketTrain(Admission|ManifestBounds|WAVBounds|SourceConfinement|RowResourcePlan|HeaderAndAssetBounds|SpecialFiles|SampleExactGeometry|StrictTensorHeader|SyntheticPreflight|TypedHeaderRejection|ModelFileStructure)$' \
  -count=1 -p=1 -parallel=1 -timeout=30s -v \
  -o /workspace/tmp/pockettrain-preflight-20260930/hardened-tests/
```

Working directory: repository root. The reviewed dependency/init closure contains CPU SIMD, half, checked arithmetic, safetensors/memory, x/sys CPU and YAML; no NVIDIA/Vulkan/purego/Cgo or model-init path. Temporary synthetic manifests/WAVs/symlinks/FIFO writes were confined to `t.TempDir`. Config uses repository-owned YAML; no released assets or dataset files were supplied.

Covered cases include early backend/mode rejection, resource ceilings, missing/malformed manifest rows, alignment/cut/token/recording bounds, noncanonical rate/channels/header, finite/full-scale PCM, path/symlink escape, known FIFO rejection, immutable inputs/no output files, deterministic report, SHA pin mismatch, strict JSON null/integer/metadata/offset admission and tiny real-file structural/hash success. Production rejects the tiny valid structure as an incomplete Pocket inventory. Full orchestration substitutes metadata through a test seam, so that happy path is not a released-model inventory check.

Compile-only focused/whole-tree vet/build and Linux arm64/riscv64 builds passed. Cross-builds were not executed. No race/full-suite, real CLI, model, codec, training or GPU run followed.

## Evidence and remaining work

Logs, exact source state, UTC metadata, test binary and source snapshot are in `/workspace/exports/pockettrain-preflight-20260930/hardened/`; the original ten-pass snapshot remains intact at the parent export directory.

Known FIFO testing does not reproduce a concurrent path replacement; there is no general adversarial-filesystem guarantee. Header and digest admission does not validate all tensor values or train a model. Actual released-model preflight, recorded-data preparation, RNG/cursor resume, gradient accumulation, update/checkpoint/export orchestration, teacher support, quality/listening and GPU backward/optimiser code remain separate work.

The source-readiness roadmap is `/workspace/notes/pockettts-local-voice-training-readiness-20260930.md`. No training data was downloaded or mutated. GPU/recovery holds remain in effect; `@default` coordinates recovery.
