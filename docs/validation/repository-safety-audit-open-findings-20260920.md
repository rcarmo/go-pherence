## Repository audit: open findings and validation limits

Every host-listed package now has a selected source-boundary review in the
[coverage matrix][coverage]. This ledger records what that review did **not**
resolve. The audit is not a line-by-line inspection of every implementation, an
exhaustive security assessment, or permission to execute an unsafe native path.
Issue [#16][issue] remains the general remediation tracker. Unavailable native
validation is filed once per platform/backend:

| Platform/backend | Tracking issue | Scope |
|---|---|---|
| NVIDIA/CUDA | [#17](https://github.com/rcarmo/go-pherence/issues/17) | CUDA capture/scratch, PTX index/scatter, Whisper device parity and direct ioctl owners |
| SpacemiT K3/RISC-V | [#18](https://github.com/rcarmo/go-pherence/issues/18) | AICPU/IME2/RVV/TCM, vendor ORT and native command validation |
| Vulkan | [#19](https://github.com/rcarmo/go-pherence/issues/19) | Physical-device parity, submissions, uncertain drain and teardown |
| ARM64/NEON | [#20](https://github.com/rcarmo/go-pherence/issues/20) | Actual ISA/runtime tests beyond foreign compilation |
| Native GGML/CGo | [#21](https://github.com/rcarmo/go-pherence/issues/21) | Shared native bounds, allocation, callback and lifecycle validation |

K3 and ARM64 link to the shared GGML issue rather than duplicating its checklist.
These issues do not authorise device recovery, service changes or frozen-evaluation
execution.

Priorities below are triage, not measured exploitability: P1 can corrupt state,
produce false validation or cross an unsafe/native boundary; P2 covers local
workload admission, diagnostics and contract gaps. "Source finding" means the
control flow or arithmetic was read, not that the fault was reproduced on a
physical device. Closing an item requires the stated follow-up, not just a host
package test.

## Native ownership and indexing

| ID | Priority / status | Source and finding | Required follow-up |
|---|---|---|---|
| N01 | P1, open; hardware blocked | `backends/nvidia/runtime`: capture stream selection is process-global; per-call driver scopes do not isolate a multi-call capture transaction. NVFP4, split-KV and BF16 scratch aliases can escape locks. Shutdown requires quiescent ownership. | Explicit owner/transaction or enforce rejection of competing work; fake-driver interleaving tests, then authorised device capture/lifetime tests. |
| N02 | P1, open; source-only | `backends/nvidia/ioctl/gpfifo.go`: notifier allocation ownership and failed token/channel rollback are incomplete. Descriptor fixes do not drain submissions or serialize mapped reads/free. | Retain every allocation/handle in an owner, reverse-order failure cleanup, fake lifecycle tests and authorised native drain checks. |
| N03 | P1, open; source finding | `backends/nvidia/ptx/q8/q8_scatter.go` uses non-atomic load/FMA/store for `dst[pos[batch],row]`; `GemvQ8_0BatchScatter` does not enforce unique positions. Duplicate destinations can race across blocks. By-work variants use atomics and are a different contract. | Specify/enforce uniqueness or implement reviewed atomic accumulation; duplicate-index device regression and numerical acceptance before deployment. No GPU execution was performed here. |
| N04 | P1, open; source-only | FP8/MLX/Q4/Q5/Q8 and NVFP4 GEMV paths use u32 linearised products after validating individual dimensions. Large accepted products may wrap before widening addresses. Some host batch helpers also multiply before checking slices. | Validate every kernel-specific total and buffer span before upload/launch; fake-buffer boundary tests, then device parity. Whisper attention/conv/mel totals were fixed separately. |
| N05 | P1, open; native dependencies absent | GGML compute/graph/quant callbacks and encoded lengths, dimension narrowing, Run/Close ownership; `llamagraph` dtype/block alignment, upload extents, post-Close decode and native allocation rollback. Its plain context is not retained with the buffer; MTP setters are placeholders. | Native-library version contract, checked byte layouts/owners and failure injection. Host shared-config tests are not a native tagged build; `ggml.h` is missing here. |
| N06 | P1, open; native dependencies absent | SpacemiT ORT trusts caller output length before `unsafe.Slice`, can retain Go input pointers through OrtValue calls, and lacks a complete Run/Close protocol. Thread-option statuses need handling. | Query output shape/dtype, own or pin retained input, serialize native lifecycle, propagate statuses and run native integration tests. |
| N07 | P1, open; K3 untested | AICPU TCM wave paths check pool-owned slices but use separate global `getTCMSlice`; a worker can return without work while the wrapper returns success. Pool joins do not own the global mapping. TCM reference counters do not prevent Close/unmap and pointers can survive Close. | Unify mapping ownership, propagate skipped-worker failure, invalidate views and prove drain before unmap on K3. |
| N08 | P1, open; source-only | RVV/AICPU single-row GEMM/pack paths have incomplete input/output extent checks. W4 quantised paths truncate unsupported row/column tails. Native RVV tags assume RISC-V implies the required extensions. AICPU env flags can select a panic C-shim stub; Q4 extraction ignores raw-source errors and block divisibility. | Shared shape/packed validators, explicit unsupported ISA/build selection, native/fallback malformed tests and K3 numerical checks. Do not replace missing ISA support with invented emulation. |
| N09 | P2, open; numerical contract | Native BF16 target is `sm_86`; emulated BF16 PTX truncates where native narrowing rounds. Exported mutable PTX variables assume process-start immutable ownership. | Document precision/target contract, validate supported architecture and prevent mutation during compilation; do not claim cross-path equality from stub tests. |

## Model and cache boundaries

| ID | Priority / status | Source and finding | Required follow-up |
|---|---|---|---|
| M01 | Fixed on host | Chunk cache and Qwen sidecar now charge payload/metadata before cloning, keep one owned state without spare KV capacity, compare tokens on lookup and serialize mutations. Zero/negative budgets disable storage; rejected replacements preserve old values. | Host exact-budget/ownership/race tests and independent review pass. Accounting is logical, not an allocator/RSS or concurrent temporary-memory cap; inputs remain caller-immutable during calls. See [remediation](repository-safety-host-remediation-20260920.md). |
| M02 | P1/P2, open; frozen source | `loader/tokenizer/{tokenizer,sidecar}.go`: unbounded whole-file/BPE work, base-vocabulary/added-token collision handling, ignored unknown symbols/IDs and normaliser fallback. Whole-piece lookup can bypass merge-rank processing. These are malformed-input and semantic-parity questions. | Independent synthetic/reference tests and a versioned migration **after** frozen-contract restrictions are lifted. Do not edit, replay or recalibrate the frozen tokenizer as audit cleanup. |
| M03 | P2, open | Legacy BERT public token/mask/sequence input, speaker ECAPA direct constructors/feature rows, VAD frame arithmetic and nonfinite timestamps assume trusted callers. Legacy agglomerative clustering still allocates an N² matrix. A representative relabel bug is fixed; its existing equal-cluster WPGMA calculation is now named accurately, not changed to UPGMA. | Checked public adapters, finite/shape/work caps and an independent clustering oracle; qualify changed numerics separately. |
| M04 | P1/P2, ownership contract | Borrowed safetensors raw bytes/shapes and expert slot views cannot outlive owner/reuse. Detach protects the owning safetensors advisor, not arbitrary independent advisors/raw readers. Hunyuan3D/model lifecycle and tensor lazy realisation still require external exclusion. | Explicit owner leases or documented/enforced caller serialization at each service boundary; mmap filesystem mutation is not prevented by path containment. |
| M05 | P2, open | Community-1 selected loaders validate exact metadata and copy weights; PCM composition is bounded and Vulkan owner gated. CPU Release still requires external exclusion, nil contexts are not universally checked, and strict trained-checkpoint/numerical gates remain unresolved. | Preserve explicit experimental status; native/device and trained-checkpoint tests before readiness claims. No new production qualification from this audit. |
| M06 | P2, open | Graph plans/nested attributes must remain immutable while live; full stride/op semantics were not audited exhaustively. Inference scheduler callbacks run under its mutex and MaxActive does not cap the waiting population. | Validate integration admission and non-reentrancy, add semantic/property tests and enforce immutable ownership. |

## Command and service boundaries

| ID | Priority / status | Source and finding | Required follow-up |
|---|---|---|---|
| C01 | P2, open | Legacy audio CLIs materialise whole audio, use external ffmpeg without uniform aggregate limits/deadlines, and permit some nonfinite or oversized chunk/worker controls. `speakercheck` now uses private temporary files, owned timeout/output capture, clamped sample offsets, finite/range controls, complete expected labels and identical JSON/text failure decisions. Whole decoded audio/disk and N² similarity work are not capped by that fix. | Private temp ownership, bounded command/media helper, finite time controls and exact expectation-length validation. These are local diagnostics, not the bounded speech-job service. |
| C02 | P2, open | Ideogram generation/VAE probes multiply height/width/channel counts after divisibility checks; very large local flags can overflow or exhaust memory. Inventory JSON is whole-file. | Checked dimensions and documented diagnostic limits. VAE smoke now closes its file, returns errors and caps grid1..64. |
| C03 | P2, partly fixed | `llmgen` now uses prepared prompt lengths and reports end-to-end throughput, not invented decode timing. Qwen flags bound steps/chunks/repeats and MiB arithmetic and its MLX helper propagates SyncErr. `llmchat`, MTP synth/smoke, whole prompt files and model-dependent prefill products still admit large trusted-local workloads. | Remaining CLI/file/model shape admission; native SyncErr execution remains in#17. Synthetic wrapper/flag tests pass without frozen model execution. |
| C04 | P1/P2, open | SpacemiT benchmark CLIs accept invalid dimensions/zero iterations and some use hardcoded FFN/graph shapes or ignore GGUF tensor errors. `spacemit_run` accepts negative tokens; native graphrun divides by unchecked head count and has a tagged Printf argument mismatch. `verifydot` constructs but never runs a command. | Host argument/metadata tests per CLI, tagged build checks with dependencies; label hint-only/scaffold tools honestly. |
| C05 | P2, partly fixed | ORT probe/export tools overwrite fixed generated files; native llama runner uses fixed output buffers/truncated diagnostics. ORT benchmark directory interpolation into Python source is fixed by argv passing; iteration/thread zero rejected there. | Exclusive/private outputs or explicit overwrite mode, bounded helper subprocess, truncation reporting and remaining iteration checks. No Python/ORT execution was used for the fix. |
| C06 | P2, open | DiffusionGemma local CLI residency GiB conversion can accept nonfinite/out-of-range values; profiling errors/Close and some fatal cleanup paths need consistent ownership. GLiNER schema/text files have no aggregate local input cap. | Checked budgets and private cleanup scopes; choose explicit supported limits rather than relying on allocation panics. |
| C07 | P1/P2, partly fixed | The older DiffusionGemma server had unbounded JSON, blocking model queues and no transport deadlines. These now have1MiB/single-document, message/token limits,429 busy and timeouts. Model execution is monolithic; GPU kernels are not cancelled by HTTP disconnect. Public auth, process-wide budgets and pre-expansion image prompt work still need deployment admission. | Keep behind authenticated deployment controls, review upstream proxy limits and native cancellation/owner boundaries. Host mock tests do not execute denoising. |
| C08 | P2, reporting contract | Empty modelcoverage family checklists now fail. Filtered top-level counts and full category/roadmap summaries deliberately remain separate outputs; zero matching category is not execution readiness. GGUF expectation-only plan and plan-error paths are fixed, but synthetic/runtime expectation combinations still need comprehensive CLI matrix tests. | Document output scopes; validate incompatible expectation/bench flags before loading. Treat metadata coverage as checklist completion only. |
| C09 | P1, fixed CLI; test distinction retained | Gemma MTP CLI no longer claims matched parity when model assets are absent; fixture-only mode is explicit and never executed/matched. Missing/invalid/nonfinite logit probes fail. Historical model tests may still exercise fixture consistency without hardware assets. | Use actual executed=true reports with preserved reference identities for numerical claims; do not equate a portable test pass with model execution. |

## Test and operational limits

Host race tests may include fixture/hardware skips. Foreign builds and fake-device
buffers establish compilation or preflight behaviour, not physical device safety.
PTX strings were reviewed at signatures/indexing/reduction boundaries; individual
assembly instructions and all launch call graphs were not exhaustively audited.
Scripts outside the covered Bun/Python checks and external services retain only
inventory coverage. UI implementation, GPU recovery and the 676/1,440 frozen
experiment are not blockers to recording this package audit, but they are not
completed by it either.

[coverage]: repository-safety-audit-coverage-20260919.md
[issue]: https://github.com/rcarmo/go-pherence/issues/16
