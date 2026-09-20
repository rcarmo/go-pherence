## Host remediation: cache admission and diagnostic correctness

The package audit left a legacy prompt cache that could clone a large state before
discovering it would not fit. Its Qwen adapter retained a second copy in a sidecar,
and neither spare KV capacity nor all metadata was represented in the budget.
Those paths now admit before copying and keep one charged snapshot. This report
records that repair and the related host-testable diagnostic fixes; the
[platform issues][findings] remain separate.

## Cache limits include the thing being retained

`runtime/kv.ChunkCache` now charges copied tokens, identity strings, layer headers,
snapshot payload and a conservative entry allowance. Qwen supplies the size of its
owned sidecar rather than storing another complete KV snapshot in the index.
K and V rows are counted independently, including unmatched V rows, together with
linear recurrent state, hidden and pre-norm vectors. Cached snapshots copy live
rows without retaining spare generation capacity. The legacy payload-only reporting
adapter is documented as such and no longer omits unmatched V rows.

Oversized insertions and replacements fail before copying and preserve previously
cached entries. Accepted inserts evict by subtraction-based budget arithmetic.
Cache mutation and Qwen sidecar publication share consistent lock ordering; reads
return independent copies and verify token equality in addition to the key hash.
Input tokens/state must remain immutable during each call. Retained accounting is
not a bound on allocator overhead, Go RSS, temporary copies or simultaneous callers.

**Compatibility change:** zero or negative cache budgets now disable storage.
The older chunk cache treated zero as unlimited. Over-budget Put now returns an
error instead of copying and then evicting the entire cache. A caller that wants
storage must provide an explicit positive budget and handle rejection. This does
not alter the newer generic prompt cache, whose zero budget already disabled it.

Exact-budget, oversized metadata/token/identity, preserved replacement, clone
independence, spare-capacity and concurrent call tests pass. A read-only independent
review found no retained alias or lock-order cycle in the new path; its caution
about the old reporting adapter was addressed. These tests use synthetic state,
not the frozen model or its results.

## Prepared tokens are part of the prompt

`llmgen` now uses the actual prepared BOS/chat-template prefix when distinguishing
prompt from generated tokens. CPU APIs already return that full prefix; the GPU
API supplies a generated suffix. The old maximum-token heuristic could include
wrapper tokens in output whenever their overhead fit inside the generation budget.
A generous-budget regression covers that case and a failed/short result cannot
be reinterpreted as generated text.

The command also stops inventing a prefill/decode split from the fraction of prompt
tokens. Without separate phase timers, it reports end-to-end output throughput,
including prefill. No new model performance measurement was taken.

Qwen's diagnostic command now validates step/draft/chunk/repeat controls, finite
acceptance/tolerance and MiB conversion bounds before cache creation or model
loading. Steps are limited to65,536, drafts to4,096, repeat count to1,024 and chunk
sizes to65,536. These are diagnostic admission limits, not a guarantee that every
permitted workload fits physical memory. Its host-download MLX LM-head helper
propagates `SyncErr`; native error handling still needs the CUDA platform tests.

## Speaker validation must compare complete results

`speakercheck` now rejects an expected-label count that differs from its complete
segment list. Speaker IDs are arbitrary cluster names, so a bijective relabelling
is accepted; invalid/nonpositive labels and empty explicit expectations are not.
JSON and text output use the same pass/fail decision. Numeric flags must be finite,
threshold is within[-1,1], context within0..30 seconds, and start/duration are
nonnegative. Sample offsets are clamped against the available extent before
float-to-int conversion.

The ffmpeg fallback uses a private temporary directory and the shared owned-command
runner, with a30-second timeout and64KiB aggregate captured-output limit. No path
is derived from a predictable process ID, and no overwrite flag is needed. Temporary
files are removed as soon as decoded samples are owned, before later `os.Exit`
paths could bypass deferred cleanup. The tests inject the WAV loader and command
runner; neither actual ffmpeg nor a model is executed. Whole audio/decoded disk
size remains a separate limit to add--bounded stderr and time do not bound output
file size or model memory.

A separate synthetic clustering regression exposed a label-update bug: after
merging a representative, changing its label altered the comparison used for
later members. The three-row fixture returned[0,0,1] despite a single merged
cluster. Capturing the source/destination labels before the loop fixes it. The
existing equal-cluster WPGMA calculation is retained and named accurately; this is
not a switch to size-weighted UPGMA or a new diarization-quality claim. Quadratic
clustering admission and legacy model input checks remain open.

## Checks and what remains

Each bounded change was tested with affected-package/importer race tests and
whole-tree vet/build before commit. The first repository sweep overlapped the
MLX helper edit and caught a temporary return-signature compile error; that run is
retained as failed and does not count as final validation. The final committed-code
NVIDIA-disabled race sweep exits zero: **110 packages pass and54 have no tests**.
Whole-tree vet/build,22 Bun tests and the339-document link/layout check pass.
ARM64/RISC-V builds pass as compilation only. Local logs are under
`/workspace/tmp/go-pherence-repo-audit/`.

Both preserved evaluation binaries, all16 freeze-manifest entries and all676
result hashes match. No predictions were inspected, rows replayed, calibration
changed, GPU recovered or services restarted. The Bun-build/Go-host UI remains an
implementation task; these fixes do not include it. Further host findings remain
in the ledger and native tests in issues#17--#21, so this closes the named repairs,
not the entire remaining plan.

[findings]: repository-safety-audit-open-findings-20260920.md
