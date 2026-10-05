## Pocket TTS resident compact embedding integration

Two opt-in scopes now use compact token gradients and compact-to-resident accumulation:

- `ptx-resident-embeddingcompact-hybrid-v1`
- `vulkan-resident-embeddingcompact-hybrid-v1`

They inherit the selected private forward-result boundaries and existing device optimiser/state checks. The private step avoids a dense embedding VJP tape temporary; dense persistent gradient/ping-pong buffers, global norm, AdamW/EMA and snapshots remain. Older scopes and public tape builders retain their previous routes and output contracts.

The source is uncommitted at parent `a90aec30567d485784f2e04d1803c543c9f2817f`. Staged inference remains eight paths and469insertions. All execution used offline CPU/mock qualification with NVIDIA/Vulkan disabled. CUDA/PTX and GLSL/SPIR-V compilation/inspection were offline only. No native GPU, released-model/audio, recovery or publication work occurred.

## Rewrite and binding contract

`compactResidentEmbeddingTape` first validates the complete ordinary dense tape and all resident bindings. It examines direct bound mode-one embedding VJP producers in sorted binding-name order. This gives deterministic added tensor/command IDs without relying on map iteration order.

For each matched binding it appends an owned compact index, a rows*width output and a mode-three command reading the same immutable seed as the original VJP. It owns the changed operand list, tensor/command/output header arrays and rebinding map. Original commands, tensor IDs, outputs and gradient roots remain present in the full source graph. The compact binding redirects only resident accumulation to the new output/index pair.

Complete compact binding/index/provenance and combined graph/byte admission then run before pruning, allocation or accumulation. Full source logical storage includes the original dense VJP output plus compact output/index, so adding compact execution does not bypass a source byte ceiling. An expanded-graph budget failure still occurs before a tape is allocated.

Pruning roots include requested outputs and all accumulated gradients/compact indices. Thus the private step discards the unused dense VJP command/output while retaining compact output/index until exactly-once resident accumulation completes. If a caller explicitly downloads the public dense gradient ID, that original VJP stays live: both representations are computed, but only the compact binding accumulates. It does not accumulate twice or reinterpret public output layout.

Graphs without a direct bound mode-one VJP return the fully admitted source and bindings unchanged. No cross-tape cache, mutable shared index or source-value rewrite is added. Unchanged nested metadata/constants stay read-only borrowed, as before. Appended arrays and result maps are independently owned, including concurrent rewrites from shared immutable sources.

The coordinator contract is documented in [compact binding validation](pocket-tts-resident-compactbindings-20261002.md); mode-three/index and accumulation formulas are documented in [compact VJP validation](pocket-tts-resident-embeddingcompact-20261002.md) and [accumulation validation](pocket-tts-resident-embeddingaccumulate-20261002.md). Sequence-order sums, parameter scale once, nonnegative training-scale admission, all binding roots, endpoint stop-gradient and terminal owner retention remain unchanged.

## Scope and replay

The new backend requires `ResidentCompactEmbeddingDriver` before resident state allocation. Missing capability returns an admission error without device allocation or tape submission. It retains the earlier single-owner graph/optimizer rules and selected private output guards.

CLI names are `ptx-resident-embeddingcompact-hybrid` and `vulkan-resident-embeddingcompact-hybrid`. Admission still requires explicit experimental-device opt-in, runtime digest and bounded logical bytes. Factory tests verify both new scopes and every component owner assignment without initializing native hardware. The new replay hash kind is `resident-embeddingcompact-backend`, bound to exact scope and caller runtime identity. Same-scope checkpoint/resume passes; restore under indexed-dense, forward-results, pruned or the other backend scope is rejected without state mutation.

Compact layout metadata is derived deterministically from admitted tokens and dimensions for each graph rather than checkpointed as mutable state. The v1 scope/hash kind and caller runtime pin identify that implementation; changing its contract requires a new compatible identity decision. Existing SIMD, attention-v1 and previous resident scope identities remain unchanged.

## Storage and setup evidence

The tiny full-step mock row recorded:

| Accounting | Forward-results baseline | Compact resident scope |
| --- | ---: | ---: |
| Tapes |11|11|
| Commands |3,709|3,709|
| Downloads |42|42|
| Executed logical tensor bytes |134,064|134,028|
| Planned storage bytes |41,984|41,948|

Gradient-download and accumulation counts remain unchanged. A recording driver inspects the actual pruned tape: both compact scopes execute exactly one mode-three embedding VJP and no dense mode-one/two VJP in the private fixture. Older scopes still execute the previous VJP. An explicit public-gradient download executes both dense and compact commands, returns the dense result numerically unchanged and accumulates compact once.

A synthetic layout at rows16/width16/vocabulary4,096 records:

| Storage contract | Dense source/path | Compact rewrite/path |
| --- | ---: | ---: |
| Full-source logical bytes |525,376|526,596|
| Pruned execution logical bytes |263,232|2,244|
| Planned tape storage bytes |263,232|2,244|

The parameter source is unused by this VJP and omitted by pruning; persistent resident parameter/gradient storage is outside the tape plan and unchanged. Full-source logical cost rises1,220bytes from the appended index/output. The pruned plan removes the dense output and retains compact output/index. These are deterministic mock storage-plan values, not native allocation peaks or released-size measurements.

At vocabulary32,768/width1,024/rows512, theoretical VJP output is2MiB instead of128MiB, but this slice does not execute that graph on hardware. Host source graph construction/admission, compact-index construction and repeated validation still cost memory/time. Compact index construction previously measured10,624B/two allocations and108.6--120.5microseconds at rows512; no new complete-row latency or native speedup is claimed. The accumulator still writes dense resident buffers and adds binary-search work per element. Cases with rows greater than vocabulary may have larger padded compact output than dense output. Selecting the explicit scope is not evidence of improvement for every shape.

## Offline qualification

Evidence directory: `/workspace/exports/pocket-gpu-training-residentembeddingcompact-20261002`.
Gate script: `/workspace/tmp/pocket-residentembeddingcompact-gate.sh`.
Final gate window: **2026-10-02 11:00:35--11:01:50 UTC**.

| Gate | Passed | Skipped |
| --- | ---: | ---: |
| Selected tests, ten repetitions |690 top-level +720subtests|0|
| Selected tests, race detector |69 top-level +72subtests|0|
| Affected CPU/mock regressions |772 top-level +1,277subtests|1|
| Compiled-artifact follow-up |69 top-level +72subtests|0|

Zero failures in final gates. The existing opt-in `TestTrainingRandomMPFRComparisonDiagnostic` remains skipped with `requires separately reviewed fixed-corpus error-characterisation gate`. It is unrun, not a passed test; no restriction changed.

New rewrite tests check original command/ID/output preservation, compact index/root dimensions, owned source/result maps/slices, numerical equivalence and zero padding, concurrent source preservation, malformed dead sources, binding inventory and combined graph/byte ceilings. Capability and expanded-budget tests verify rejection before resident/tape allocation. Actual dispatch and explicit public downloads test pruning safety and exactly-once accumulation.

Both scope labels pass pinned PyTorch complete-step/all-gradient/update/EMA references, independent CPU objective weight and clipped-update tests, padding, repeated/absent tokens, empty text and voice, seeded nonaccumulating and zero-seed accumulating conditioner calls. Isolated token-gradient tests inspect the admitted dense resident gradient after compact accumulation against the legacy reference. Synthetic zero branches complete only that isolated mock test; they are not production missing-gradient defaults.

Checkpoint/resume and cross-scope restore, all-tape failure injection, selected-output shape/nonfinite/alias/overflow guards and live-owner checks pass. Compact preflight, execution and index-output errors reject trainer/run publication, hold resources and block retry. Existing complete-source admission, borrow guards, clipping and full snapshot ownership remain in the regression gate. Numerical tolerances are unchanged; generated-output hashes/byte identity are not used as signal acceptance gates. Exact replay comparisons cover their structural contract only.

The new scope predicate has100% scoped coverage and rewrite97.6%; the index-build error is defensive after complete dense source admission, not evidence of an invalid-ID bypass. Package coverage is89.3% training,12.9% NVIDIA runtime,36.7% Vulkan,68.8% Pocket TTS and14.9% pockettrain under this selector. This is not whole-backend/full-device coverage.

Vet, host `go build ./...`, Linux ARM64/RISC-V pockettrain cross-builds, formatting, model-layout and diff checks pass. Offline CUDA12.8 sm80/PTX8.7 `--fmad=false`/ptxas, GLSL/spirv-val and conservative reflection pass. This integration changes no kernels beyond the earlier primitives. Development errors were not required to pass the final gate; logs retain the initial successful narrow checks and final evidence.

## Review limits and remaining work

An invariant judge described the integration as conceptually sound, while emphasizing identical seed snapshots, capability checks before allocation, name-based bindings, dense-prune reachability, public dual-compute without double accumulation, identity separation and whole-pass preflight. The same seed tensor is used; complete admission/name-based coordinator/pruning/public-output/accounting/resume/failure tests cover those conditions. The judge did not inspect source or execute native code; no independent source/native approval is claimed.

Persistent dense gradient/optimiser buffers and work remain. Transformer/flow audio-hidden handoffs, prediction admission, EOS scalar/diagnostic synchronisation, full snapshots/live host copies and authorised native resources/performance/voice qualification are still open. These remain resident hybrids, not full-device training. Native execution/recovery, listening and publication holds are unchanged; this integration does not complete the overall implementation goal.
