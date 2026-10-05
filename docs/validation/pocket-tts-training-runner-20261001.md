# Pocket TTS CPU training workflow — 1 October 2026

Added an experimental Linux SIMD workflow for recording preparation, training, checkpoint resume and inference export. No trained voice has been validated. Separately named experimental Vulkan/PTX attention-backward and flow hybrids now extend the CPU workflow; complete GPU training is unfinished.

Base `a90aec30567d485784f2e04d1803c543c9f2817f` plus previously held inference/preflight/run-state/bundle/sample candidates. Rui requested removal of cross-agent timeline messages and direct completion; no further agent approval loops were used.

## Implementation

- `cmd/audio/pockettrain/prepare.go`: pinned local config/model/tokenizer/aligned manifest preflight, recording root confinement, same-descriptor recording/model admission, frozen Mimi encoding, target normalisation, explicit one-frame EOS padding and bounded exclusive JSON publication with provenance.
- `model/pockettts/training_runner_linux.go`: owned prepared rows, experimental sampler opt-in, one-row SIMD updates, global gradient clipping, AdamW/EMA, run-state commit after successful update, one reusable current-shape workspace, failure poisoning and fully decoded bundle resume.
- `cmd/audio/pockettrain/train.go`: resource/row/provenance admission, exclusive process-writer lock, seeded independent streams, progress JSON, immutable-generation checkpoints, optional inference safetensors export.
- New tests exercise tiny actual training updates/cancellation/save-decode-resume/corruption/uncertain-update stopping, CLI rejection, prepared JSON and latent normalisation. No released-size model, real recordings or GPU workload is executed by these tests.

Existing read-only preflight stays the default mode. Training requires `-allow-experimental-sampler` and an explicit full build source revision. Unsupported GPU requests fail before file access. Complete usage is in `cmd/audio/pockettrain/README.md`.

## Limits

The random adapter has fixed-corpus MPFR characterisation, not production-wide numerical/distribution/voice-quality acceptance. The runner requires explicit experimental opt-in. It uses always-on distillation, zero dropout, one row/update, constant LR and no accumulation; it does not silently reproduce upstream probabilistic skipping/default recipe.

Prepared targets must fit374valid codec frames plus one existing padded frame, voice≤62frames, tokens≤512. Corpus JSON, sample draws and checkpoint bytes have explicit ceilings. Training planner and additional snapshot/corpus estimates are conservative admission estimates, not measured peak RSS or free-disk guarantees. Preparation retained-memory limits do not establish encoder peak-memory guarantees.

Same-descriptor admission protects path replacement, not hostile in-place writes. Inputs must remain immutable during execution. Prepared row/model/config identity is pinned; run identity remains declared numerical/options provenance, not proof every platform property is identical. A stale directory lock after process death needs operator disposition.

Failures before parameter update abandon the proposal. Failures/invalid numerics after possible Adam/EMA mutation poison the runner and require fresh coherent checkpoint recovery; no rollback/retry is attempted. Bundle-visible/durability-uncertain publication errors remain typed and old generations are preserved. No power-crash proof or independent listening acceptance is claimed.

## Verification

Bounded tests run in fresh `env -i`, offline local Go, `GOMAXPROCS=2`, both NVIDIA and Vulkan disable flags, nice10, serial execution and inner/outer deadlines. Eight new synthetic CPU tests passed ten times each (80named passes), plus focused race tests, vet, host and Linux arm64/riscv64 builds. Final verified window:2026-10-01 06:37:47.243376824–06:37:58.641111243 UTC, exit0. The source/checkpoint layout gate also passed (Go docs layout tests plus5Bun tests). Logs, source snapshots, binaries, metadata and exact test controls are archived at `/workspace/exports/pocket-training-runner-20261001/`. Earlier passing intermediate gates and one capability-field compilation failure are preserved separately; final source uses the actual `Capabilities.Arch` field.

Host binary, Linux arm64 and riscv64 builds are produced outside the repository. Cross-building does not validate foreign execution. No model/recording/trained-voice listening run occurred, and no GPU recovery/probe was performed. Existing staged eight-file inference candidate remains separate, uncommitted, awaiting listening/publication disposition.

Cross-agent cleanup deleted128agent-message rows from this chat without cascading into reports or user messages. Original agent text was archived in `/workspace/exports/pocket-cross-agent-timeline-cleanup-20261001/`; report/evidence attachments were retained. The initial delete rate-limit was handled through bounded backoff, and zero matching cross-agent rows remained after verification.

## Follow-up correctness and performance audit

At Rui's07:24goal update, hardened the new CPU workflow without changing earlier inference/sample candidates:

- Decode prepared corpus only once; charge mapped model plus corpus/training/snapshot admission together.
- Validate one isolated proposal during corpus admission, eliminating repeated full-permutation clone/validation per row. Row-dependent checked draw counts remain enforced.
- Reject nil/malformed trainers, moment/EMA shape mismatches and EMA options that round to1inF32; keep failed-update recovery explicit.
- Replace per-parameter `Hypot` calls with compensated F64 squared-norm accumulation and one sqrt. FiniteF32 squares and int-sized counts fitF64. Independent3–4–5 and large-finite clipping tests passed; no end-to-end speed claim follows.
- Admit EOS-padded geometry before codec calls, verify codec/preflight frame/channel agreement and own codec/token buffers. Synthetic codec tests exercise prompt/target sample ranges, exact normalisation, EOS padding and ownership without loading a model.
- Publish inference exports through a private scratch directory and exclusive link, preventing a raced destination from being overwritten. An existing-file preservation test passed.

Final bounded hardening gates:2026-10-01 **07:34:34.297682019–07:34:44.183364419 UTC**, exit0.14new CPU tests repeated10times (140named passes), focused race tests,62integration tests plus282subtests (no skips/failures), focusedvet and host/Linuxarm64/riscv64 builds passed. Integration included existing tiny loss/gradient/binary-checkpoint/run-state/bundle/sample tests; it explicitly excluded the fixed-corpus MPFR opt-in diagnostic and all released-model opt-ins. Evidence/source/binaries are separate in `/workspace/exports/pocket-training-runner-audit-20261001T0730Z/`; old archives are unchanged. The read-only independent audit delegate timed out and supplies no approval.

## Completed-update boundary audit

The next pass changed only `cmd/audio/pockettrain/train.go`, its tests and documentation. Due and final checkpoints now publish before progress output. Progress includes `checkpoint_step`, so unsaved interval updates are explicit. Cancellation and broken output each save a dirty completed boundary once; a failed update is never saved. Checkpoint/generation failures stop without retry, preserving typed publication-uncertainty errors and joined cancellation/output causes.

Training also enforces its declared preparation policy before model loading: exactly one zero EOS-padding row, non-empty bounded voice/text conditioning, finite latents, matching latent lengths and in-range tokens. The lower-level runner's explicit caller-provided mask policy is unchanged.

The final new-test gate ran **07:49:14.623673669–07:49:17.246420557 UTC**, exit 0, on 1 October 2026. Ten boundary/admission tests passed ten times (100 top-level passes and 180 subtest passes), plus focused race checks, vet and a host build. `runPreparedUpdates` and `validatePreparedRow` each have 100% statement coverage under that selector; the package total is 6.1% because these tests deliberately exclude model-loading and preparation paths. The generation function's OS entropy source is not fault-injected; generation failures are tested through an explicit callback. Earlier 07:34 gates were not repeated.

Restore rejection was then tested against decoder-valid, coherently published bundles with mismatched optimiser settings, EMA decay, negative second moments, buffer shapes and corpus row counts. All five cases preserve the fresh trainer's parameters, buffers, moments, EMA and run metadata; that runner can still train after rejection. This new test passed ten times (10 top-level and 50 subtest passes) plus race checks in **07:51:47.652475985–07:51:53.513340915 UTC**, exit 0. No production restore code changed.

Evidence is in `/workspace/exports/pocket-training-boundary-20261001T0742Z/`, including source, test/race logs, coverage and a host binary. A read-only restore/admission delegate timed out after 120 seconds and returned no review. No released-model, recording, listening or GPU run occurred. The inference index and older evidence were left unchanged.

Remaining project work: released-size resource/timing validation, real voice corpus and training-quality evaluation, statistical sampler qualification, and full-device orchestration/residency/checkpointing. Vulkan/PTX primitives and separately named hybrid integration are covered in [the GPU implementation record](pocket-tts-gpu-training-primitives-20261001.md). The overall SIMD+GPU voice-training objective is not complete.
