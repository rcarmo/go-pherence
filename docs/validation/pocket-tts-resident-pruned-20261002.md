## Pocket TTS resident tape dependency pruning -- 2 October 2026

`ptx-resident-pruned-hybrid` and `vulkan-resident-pruned-hybrid` omit dead execution commands/tensor allocations and pure forward derivative downloads from resident-step training. They preserve retained command order and arithmetic. Global norm, dispatch/boundary/weighting-input assembly and completed-update full host snapshots still use CPU work. Native numerical/resource/performance qualification has not run.

Parent: `a90aec30567d485784f2e04d1803c543c9f2817f`. Source uncommitted; staged inference index unchanged at eight paths/469 insertions. No native GPU/runtime initialisation/probe/launch/recovery, real-model, audio or listening run occurred. Execution/listening/publication holds remain.

## Dependency closure and inventory

`PruneTape` first validates the complete original SSA tape against its logical byte ceiling. Malformed dead commands, nonfinite dead constants and over-budget source graphs reject before pruning. Every requested root must be an original declared output. A reverse walk marks each producing command and all its operands; compact tensors keep ascending original ID order, commands retain original order and fields. Shapes/command operands/output metadata are copied. Constant Values remain read-only references into immutable source storage. No reassociation, constant folding, in-place reuse or cross-tape cache is added.

The remap contains -1 for every removed ID. Constant-only roots produce a validated zero-command tape: upload/download with no dispatch. Uninitialised constant-only outputs still reject. Native adapter mocks test constant tapes with no launch/plan and correct owned cleanup. Earlier prototype kept the full graph for constant roots; focused review prompted removal of that unnecessary execution path.

`RunPrunedTape` performs complete binding admission and resident name/extent inventory checks before pruning, including unused/dead parameters. Its roots are requested downloads plus every named gradient when accumulation is enabled. Borrowed current parameters are mapped only if live. An accumulated gradient can be independent of its parameter value (for example bias VJP); that gradient still maps/accumulates by resident name even when the parameter input disappears. Full-source bindings remain the acceptance inventory. Tests cover dead unknown parameters, live independent gradients, shared fanout, reshape dependencies, duplicate/multiple roots and command ordering.

Active borrow, complete-pass accumulation preflight, terminal owner retention, no uncertain retry and output finite/shape validation remain. Admission failure closes a safe tape without mutating optimiser state; submitted/download/close/accumulation failures retain owners. RunPrunedUnboundTape shares these rules without parameter mutation. Unpruned APIs remain available and old CLI scopes select them.

## Forward boundaries

The new model scope requests only value outputs (plus flow time-JVP) when accumulation is disabled and both seeds are zero. It returns new host-owned zero derivative buffers to existing boundary consumers. Forward flow, transformer, weighting, projection, audio assembly and sequence assembly therefore skip reverse graph execution and gradient downloads. Nonzero seeds retain all requested derivatives even without parameter accumulation; minimal-LSD endpoint input VJP is explicitly tested. Backward accumulation roots include all parameter gradients regardless of requested downloads. Utility tapes are pruned too.

Separate `*-resident-pruned-hybrid-v1` replay identities preserve earlier graph/launch/error surfaces. Pure forward mode can omit errors from dead arithmetic such as zero times an overflowing derivative; retained outputs and update finite checks still enforce the new scope's live boundary contract. Full original shape/constant/kernel-op syntax and byte ceilings are admitted. Dead native kernel availability is not preflighted after pruning because those commands will not execute.

Host graphs, parameter copies, zero buffers and constants are still materialised before this pass. Pruning reduces the graph received by the driver; host construction overhead, dense embedding VJP, repeated forward/backward graphs, snapshots and inter-tape transfers still require work. No native kernel source/module changed.

## Offline evidence

Tiny complete step under identical CPU mocks: commands **5746→3891**, tensor specs **7244→4996**, summed logical tape storage **212136→142880 bytes**, boundary tensor downloads **148→116**. Tape-call count and parameter accumulation count stay unchanged. These totals are sums across calls; live peak allocation, RSS, transferred bytes and latency are unmeasured. Pruning has no native speedup acceptance.

Pinned complete-step metrics/loss3e-5, all-gradient/input5e-5, parameter/EMA updates4e-6 stay unchanged. Nine objective variants and padded-row parity pass. Both named scopes pass checkpoint/resume parameters/M/V/EMA/raw replay with stable resident allocation count and cross-scope rejection. Every complete-step construction position is failed in turn; unchanged host/replay, no retry/checkpoint and retained owner. Bound/unbound failure tests cover construction/nil/run/output/download/count/extent/nonfinite/close/admission/accumulate, as applicable. Constants alone dispatch no work in PTX/Vulkan mocks; source admission still rejects malformed dead work.

A focused self-contained delegate reviewed the algorithm and flagged the constant-only full-graph fallback's identity/liveness semantics. The final implementation removes the fallback and preserves -1 dead mappings. Added constant-only/inventory/native-adapter tests pass. The delegate did not inspect final source or native numerics; no independent final approval.

Final gate: **01:07:13.294272709--01:08:15.543815318 UTC**, exit0. Twenty-one tests repeated ten times: **210 top-level+300 subtest passes**; race **21+30**; regressions **41+38**. No failures/skips.

Local Go1.26.3, GOENV off/empty GOFLAGS, GOPROXY/GOSUMDB off, GOMAXPROCS2, both GPU-disable flags, nice10, serial packages/tests,30-second inner/120-second outer deadlines. Affected backend/model/CLI vet, host build, ARM64/RISC-V CLI cross-builds, layout and whitespace gates pass. Foreign binaries not executed. No kernel compilation needed because kernel sources unchanged.

Scoped PruneTape97.5%, resident runTape92.7%, model pruning/scope/helpers100%; selected package20.7%. Native execution coverage absent. Evidence: `/workspace/exports/pocket-gpu-training-pruned-20261002/`.

Remaining implementation: all-device global norm and fused boundary/weighting-input orchestration; reduce host graph/constants, live intermediate allocations, dense embedding VJP and host snapshot/transfer churn. Native trig/loss/released-size numerics/resources/performance and sampler/new-voice/listening qualification await coordination.
