## Pocket TTS resident conditioner projections and EOS -- 1 October 2026

`ptx-resident-projection-hybrid` and `vulkan-resident-projection-hybrid` now execute speaker/audio projections and the EOS affine head on resident tapes, alongside flow, transformer, weighting and AdamW/EMA. Token embeddings, BOS/shifted sequence assembly, scalar losses, time embeddings and final global norm-summary reduction stay on CPU. Complete host state synchronises every update. Full-device training and native qualification are unfinished.

Parent: `a90aec30567d485784f2e04d1803c543c9f2817f`. Source is uncommitted; the held inference index is unchanged at eight paths / 469 insertions. No native runtime/probe/launch/recovery, real model, recording or listening run occurred. No kernels were added or changed in this slice.

## Projection tapes

`training_projection_tape.go` builds one row-major affine forward and input/parameter VJP from existing dual linear primitives with zero tangent inputs/seeds. Bias is optional. Weight names are explicit because `flow_lm.speaker_proj_weight` differs from normal `.weight` naming; audio input uses `flow_lm.input_linear.weight`, EOS uses `flow_lm.out_eos.{weight,bias}`. Shapes, exact lengths, owned F32 linears, names, finite values and complete logical tensor storage are checked before copies/zeros. Kernel limits are rows<=1024 and input/output dimensions<=16384.

`flowlm_training.go` captures the projection backend for paired forward/backward. Batched speaker/audio values return to CPU sequence assembly; the shifted audio layout is unchanged: BOS then target latents excluding the final frame. EOS values return after transformer forward. Seeded EOS input derivatives combine with the flow conditioning derivatives before transformer backward; speaker/audio input derivatives feed raw voice latents, BOS and the prior target frame. Text embedding gradients still accumulate on CPU. Projection parameter gradients stay resident and their host arrays remain zero.

An empty voice reference performs no speaker forward, but seeded backward executes one zero input/seed row to materialise an explicit zero speaker-weight gradient. This avoids a host zero array/upload proportional to the complete projection weight and satisfies the resident optimiser's complete-gradient admission. There is no implicit skipped-parameter update.

The projection modes extend resident weighting under separate `*-resident-projection-hybrid-v1` replay identities. Every projection call verifies the current model/component owner and explicit speaker/input/EOS selection. Original resident and resident-weight v1 retain their previous CPU projection arithmetic; backend injection or cross-scope restore rejects before mutation. Logical resident/tape ceilings are separate, with combined peak up to twice the CLI ceiling before native overhead and host graph/snapshot memory.

## Offline checks

An independent small integer matrix-calculus fixture covers affine output, dInput, dWeight and dBias for bias on/off. Tests also cover malformed topology, names, byte ceilings, finite inputs/parameters/results and disjoint/owned boundary values. Full-step host interpretation checks all accumulated parameter gradients against the pinned independent PyTorch fixture (`training_step_pytorch.json`, upstream `0acce6b2f390150267557770d2098c5caa9a18ac`) at unchanged 5e-5, loss at 3e-5 and parameter/EMA updates at 4e-6.

For the tiny two-valid-row fixture, four projection parameters accumulate exactly once (speaker weight, input weight, EOS weight/bias); forward calls do not accumulate. Projection mode downloads 116 boundary tensors per step (100 prior flow/transformer, four weighting and twelve projection boundaries), without parameter-gradient downloads. Voice/audio input gradients and CPU BOS/embedding branches match the pinned fixture. Both named scopes pass checkpoint/resume for parameters, moments, EMA and raw replay, with stable resident allocation count.

Failure tests cover speaker/input/EOS forward and backward tape admissions, changed/removed projection owners, malformed explicit calls and cross-scope restore. Failed steps leave the host trainer and replay unchanged, retain active/uncertain owners and block retry/checkpoint. Empty voice is tested through an actual tiny runner update and checked for zero speaker gradients. The read-only review delegate timed out after 60 seconds without findings or approval.

Final gate: **23:32:37.135735214--23:33:15.630376449 UTC**, 1 October 2026, exit 0. Ten tests repeated ten times gave **100 top-level and 20 subtest passes**; race gave ten top-level and two subtest passes. Affected regressions gave **63 top-level and 70 subtest passes**. No final failures or skips.

Controls: local Go1.26.3, `GOENV=off`, empty `GOFLAGS`, `GOPROXY=off`, `GOSUMDB=off`, `GOMAXPROCS=2`, both GPU disable flags, nice10, serial packages/tests, 30-second inner / 120-second outer deadlines. Focused vet, whole-tree host build, Linux ARM64/RISC-V CLI cross-builds, layout (Go docs tests plus five Bun tests / 3,008 assertions) and whitespace checks pass. Cross-built binaries were not executed.

Scoped coverage: projection builder 95.7%, boundary validation 100%, resident projection ownership/scope checks 100%, evaluation 95.2%. Selected package total is 26.0%. Native arithmetic/transfer/dispatch/cleanup, released-size resource use and latency, sampler distribution and trained-voice quality are unmeasured.

Archive: `/workspace/exports/pocket-gpu-training-projection-20261001/`. Remaining device completeness is embedding lookup/gradient reduction and sequence assembly, scalar loss/time-embedding orchestration and all-device global norm. Per-call graph/activation allocations and complete host synchronisation remain baseline costs requiring later measurement.
