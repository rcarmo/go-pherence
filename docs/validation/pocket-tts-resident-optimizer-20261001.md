## Pocket TTS resident optimiser state -- 1 October 2026

A bounded resident optimiser coordinator now owns parameter, gradient, branch, moment and EMA buffers through explicit PTX/Vulkan drivers. It accumulates named host-supplied gradients on device, downloads norm summaries, applies AdamW/EMA, validates mutable state and exports owned snapshots only at completed boundaries. The training runner is not yet connected to this coordinator; full-device training is unfinished.

Parent: `a90aec30567d485784f2e04d1803c543c9f2817f`. Work is uncommitted. The inference index stays eight paths / 469 insertions. No native runtime initialisation, probing, execution, recovery, real model, recording or listening run occurred.

## State and reduction

`backends/training/resident.go` accepts a completely validated snapshot: named parameter/M/V/EMA vectors, configuration and completed step. Zero-step absent moments are materialised by the model converter; EMA-disabled runs still own distinct scratch. Each parameter owns eight tensor-sized buffers (parameter, gradient ping-pong pair, branch, zero, M, V and EMA) plus a three-value summary per 256-element chunk. Logical admission is `32*n + 12*ceil(n/256)` bytes per parameter. Native allocator alignment/metadata, kernels/plans, host snapshot/branch storage and peak RSS are separate.

All branch names, lengths and values are checked before upload. Each update requires every named gradient explicitly supplied; missing parameters never receive implicit zeros. Deterministic sorted parameter order controls upload/update sequencing. Ping-pong accumulation keeps writable outputs disjoint from inputs. The same native buffers persist between successful updates; graphs do not yet bind their parameter inputs to these resident allocations.

Norm kernels emit `[maximum_abs, scaled_square_sum, invalid]` for chunks of at most 256 values. Scaling avoids square overflow for finite F32 input, including MaxFloat32. Compensated F32 chunk sums and compensated F64 host combination compute the global norm; the final clipping scale and AdamW bias corrections are host constants. The first uncompensated test at 255 elements exceeded the fixed 3e-7 relative bound; compensated sums pass without a wider threshold. This is a device reduction plus small host reduction, not an entirely device-computed global norm.

The optimiser checks every update range and constant before any parameter mutation. Post-update summaries validate finite parameters/M/V/EMA and nonnegative V before advancing the completed step. Disabled EMA scratch remains valid. Snapshot requires a clean boundary and returns copied host arrays; in-place restore is unsupported. Construct a fresh owner from a coherent snapshot.

Any allocation, upload, dispatch, download, numerical-validation or failed-close error conservatively poisons the owner and retains all resources. A nonnil owner returned with a construction error must stay reachable. Pure update range preflight can reject without mutating state. Uncertain failed updates cannot retry, snapshot or close normally; recovery requires explicit operator/process disposition and a fresh coherent checkpoint. This policy may retain allocations until process exit and does not establish crash durability or leak-free recovery.

## Native wrappers and checkpoint conversion

`backends/nvidia/runtime/training_resident.go` wraps owned buffers and existing vector/AdamW kernels, with a new norm-summary kernel. PTX norm failures poison the operator. `backends/vulkan/vulkan_training_resident.go` records each accumulation/norm/update stage in an owned plan; failed submitted work retains that plan and blocks further calls. These are synchronous baseline paths with per-operation plan creation; no performance acceptance exists.

`model/pockettts/training_resident_state.go` converts validated full checkpoints to resident initial state, including explicit zero moments at step zero, and joins completed snapshots with immutable frozen buffers. Reverse conversion binds exact parameter inventory, optimiser configuration/EMA policy and nondecreasing step. It validates the full result before returning it; no partial live trainer mutation occurs. Existing bundle publication still needs a device-aware runner integration.

## Offline tests and limits

Host drivers test actual coordinator sequencing, explicit branch accumulation, a 3-4-5 clipped update, snapshot ownership, fresh-state resume, missing gradients, full-pass preflight, partial allocation, failed uploads/dispatches/downloads/close, post-update invalid state and terminal rejection. Checkpoint conversion covers EMA on/off, initial moments, finite state, nonnegative V, wrong names/configuration and preservation of live trainer state on rejection.

The coordinator's host AdamW schedule matches the pinned independent PyTorch fixture (`training_step_pytorch.json`, upstream `0acce6b2f390150267557770d2098c5caa9a18ac`) across 1,212 parameter/EMA values: maximum error **1.1920928955078125e-7**, RMS **6.262139032995525e-9**, below the unchanged 4e-6 optimiser tolerance. Host norm summaries match independent F64 squares at lengths 1/255/256/257/1025 with 1e20 inputs and a fixed 3e-7 relative bound. Native accuracy/performance has not been measured.

Backend tests replace all CUDA/Vulkan functions with host mocks. They check kernel ABI, pointers, cancellation, extents/aliases, buffer transfers and failed-submission owner retention. The mock transfers and launches do not execute shader arithmetic. A Vulkan shader initially emitted an unadmitted logical instruction; nested conditions kept the existing closed SPIR-V envelope unchanged. That failed test log is preserved.

The final main gate ran **22:37:00.917895523--22:38:00.485921056 UTC**, 1 October 2026, exit 0. Thirteen tests repeated ten times gave **130 top-level and 70 subtest passes**; race gave **13 top-level and seven subtest passes**. Affected regressions gave 45 top-level and 58 subtest passes. Two added boundary/inventory tests passed ten times plus race in **22:40:19.712360110--22:40:31.384975074 UTC**. No final failures or skips.

Controls: local Go 1.26.3, `GOENV=off`, empty `GOFLAGS`, `GOPROXY=off`, `GOSUMDB=off`, `GOMAXPROCS=2`, both GPU disable flags, nice10, serial packages/tests, 30-second inner / 120-second outer deadlines. Vet, whole-tree host build, Linux ARM64/RISC-V CLI cross-builds, layout (Go tests, five Bun tests / 3,008 assertions) and whitespace checks pass. Cross-builds were not executed. GLSL/SPIR-V compile/validation and CUDA12.8 PTX8.7/sm80 compile/assembly pass offline, with FMA contraction/FTZ disabled.

Scoped coverage: resident construction, accumulation, snapshot, close, norm admission/summary helpers 100%; byte admission 96%, update 87.5%. PTX norm/update entrypoints 100%, driver helpers 84--100%; Vulkan norm stage 100%, driver methods 64.3--100%. Full-state conversion is 90.9% into resident and 95.7% back to full state. Selected package total is about 8%; native runtime success/error coverage and lower-level recovery are open. The review delegate first failed its workspace-path check, then a read-only retry timed out after 60 seconds; no independent review or approval was obtained.

Archive: `/workspace/exports/pocket-gpu-training-resident-20261001/`. Next implementation work is binding graph parameters/gradients to resident state and integrating completed updates/snapshots with runner replay and immutable checkpoint bundles. Native numerics/resources/performance, real voice quality and publication require separate authorisation.
