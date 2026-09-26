# JevBench v1.4.0: complete MoJev accuracy with GSO input parity

## Full public-set result

**MoJev:137/231 correct (59.31%). Historical GSO:196/231 (84.85%).**
Both return231 strict-valid answers, with zero failed requests or input refusals.
GSO leads by59 answers, or25.54 percentage points.

This supersedes the earlier coverage-limited diagnostic runs as the full public
accuracy comparison. All items count; there is no supported-subset denominator.
It is not an official JevBench sealed-set score or ranking.

| Tier | Items | MoJev correct | Historical GSO correct |
|---|---:|---:|---:|
| Easy | 48 | 42 (87.50%) | 48 (100.00%) |
| Original | 72 | 49 (68.06%) | 71 (98.61%) |
| Hard | 111 | 46 (41.44%) | 77 (69.37%) |
| **All public** | **231** | **137 (59.31%)** | **196 (84.85%)** |

### Question type

| Type | Items | MoJev correct | Historical GSO correct |
|---|---:|---:|---:|
| Choice | 139 | 85 (61.15%) | 121 (87.05%) |
| Noul | 74 | 41 (55.41%) | 61 (82.43%) |
| Score | 18 | 11 (61.11%) | 14 (77.78%) |

The pinned benchmark uses argmax correctness for Score as well as the other
types; expected ordinal values are a separate metric. No scoring rule changed.

### Family

| Family | Items | MoJev correct | Historical GSO correct |
|---|---:|---:|---:|
| adequacy | 12 | 5 (41.67%) | 11 (91.67%) |
| adversarial | 6 | 4 (66.67%) | 6 (100.00%) |
| ambiguous | 7 | 1 (14.29%) | 6 (85.71%) |
| extraction | 24 | 21 (87.50%) | 24 (100.00%) |
| fact | 12 | 6 (50.00%) | 12 (100.00%) |
| intent | 24 | 18 (75.00%) | 24 (100.00%) |
| judge_hard | 17 | 7 (41.18%) | 12 (70.59%) |
| long_policy | 19 | 5 (26.32%) | 14 (73.68%) |
| multi_hop | 18 | 5 (27.78%) | 14 (77.78%) |
| ordinal | 12 | 10 (83.33%) | 12 (100.00%) |
| policy | 12 | 8 (66.67%) | 12 (100.00%) |
| probability | 10 | 7 (70.00%) | 8 (80.00%) |
| routing | 12 | 11 (91.67%) | 12 (100.00%) |
| routing_hard | 5 | 5 (100.00%) | 5 (100.00%) |
| temporal_numeric | 15 | 4 (26.67%) | 3 (20.00%) |
| tool_selection | 12 | 12 (100.00%) | 12 (100.00%) |
| tradeoff | 6 | 2 (33.33%) | 2 (33.33%) |
| trap | 8 | 6 (75.00%) | 7 (87.50%) |

Paired outcomes:126 both correct,11 only MoJev correct,70 only GSO correct,
24 both wrong. Small family counts should not be treated as precise estimates
of broader domain ability.

### Distribution metrics

Pinned upstream aggregates, same231 valid distributions:

| Metric (lower is better) | MoJev | Historical GSO |
|---|---:|---:|
| Mean Brier score | 0.57764 | 0.26735 |
| Expected calibration error | 0.14500 | 0.12888 |
| Ordinal MAE (Score items) | 0.57856 | 0.32961 |

These are public-corpus observations, not held-out calibration qualification.

## Why this comparison has complete input coverage

The two earlier runs were diagnostic:

-512-capacity text-only:103 correct,141 valid,90 refusals.
-4096-capacity text-only:122 correct,196 valid,35 structured-state refusals.
-**Current4096-capacity compatibility path:137 correct,231 valid,zero refusals.**

The [CPU context extension](mojev-cpu-context-20260926.md) removes the512 limit
without truncation. The [structured-state compatibility decoder](mojev-systemone-state-20260926.md)
matches pinned GSO compact-JSON state rendering while leaving MoJev's strict
text-only decoder unchanged.

Before inference, the actual GSO request compiler and state renderer were invoked
on all231 archived requests through test overlays. Every request compiled, and
MoJev's rendered state matched byte-for-byte. Question validation, reserved-token
checks and token capacity checks also passed. Maximum logical request length was
4065 tokens; maximum branch length3930, below4096. After inference, request
state/questions matched archived GSO requests exactly except for the model ID.

This establishes input/answer-contract parity **for this corpus**, not every GSO
API shape. Structured criteria/instructions remain outside the MoJev decoder's
scope; none is required by these231 tasks. Additional duplicate-key/depth/size
safety checks remain stricter. Each model keeps its own prompt renderer and
tokenizer, so final prompt tokens are not identical. Parity does not mean forcing
the same predictions.

## Pins and collection

- JevBench v1.4.0 suite: `2fa63fa3226cb369795525ed011800f57dcbd894`.
- Fresh MoJev implementation: `7d414b99b6a419840f2347518dc430036738f42a`.
- MoJev checkpoint: `0c8695b6252f4205907433d4e196a94f032e60c3`, original0.8B
  weights, native repaired-policy F32 SIMD; same four asset hashes checked on load.
- `GOMAXPROCS=6`, CPU capacity4096, NVIDIA disabled; Go1.26.3/Linux amd64,
  reported Intel i7-12700. One checkpoint process at a time.
- Historical GSO service: `b18ee0d4748bac436999aa72c000e06406c3cce6`,
  Gemma4-12B IT UD-Q4_K_XL, RTX3060 12GB. Archived exact responses, hashes and
  upstream aggregates were re-audited offline. No GSO model was rerun.

The benchmark uses the unmodified pinned TypeSafeAdapter, Runner and scorer via
a temporary loopback-only MoJev transport. No answer/label tuning, prompt
truncation, omitted items, probability repair, warmup or successful-item retry.
The original HTTP wire requests stay unchanged; structured state rendering is
an explicit, tested implementation feature rather than an ad-hoc harness edit.

**Collection was interrupted after200 durable results.** The server and runner
were stopped by the external tool interruption. The200 records and raw hashes
were verified and preserved unchanged; there was no orphan raw response. The
same exact server executable was restarted for the remaining31 items. One
unfinished in-flight request without a durable response was reissued. No
completed item was run again. Final231 IDs are unique and complete, and the
pinned upstream summary was recomputed exactly from the durable records.

This is a two-segment collection, not an uninterrupted run. The first segment's
final process RSS/host telemetry was not flushed before interruption; its request
and outcome records were fsynced. The resumed segment's logs and exit0 are
preserved. Runtime/cache/host conditions differ across the restart, limiting
performance interpretation but not changing the counted task outcomes.

## Latency and resource observations

| All231 items | MoJev CPU | Historical GSO GPU |
|---|---:|---:|
| HTTP p50 |875ms |374ms |
| HTTP p95 |31,156ms |15,751ms |

Unlike the diagnostic runs, these distributions contain no quick refusals in
place of difficult tasks. They still do **not** establish controlled relative
speed: model sizes, precision, hardware, collection dates and startup histories
differ. One attempt per item is not a repeated latency study. Historical GSO
excluded its thermal preparation waits from decision latency.

The resumed MoJev segment took4m55.09s wall including load, with peak RSS
4,990,976KiB, zero swaps and final exit0. This is a **segment peak**, not a measured
whole-run peak. No fresh GSO memory data was collected. Costs remain unknown/null;
local execution is not assigned a fictitious zero compute price.

## Audit, limits and completion

Offline audits verify unique231-task coverage, exact requests except model ID,
raw-response hashes, probability extraction, pinned scoring and final aggregates.
All231 answers are strict-valid; none requires renormalization. A delegated judge
review of the reporting arithmetic/protocol found no blocking issue with this
public reproduction, subject to the historical-GSO, two-segment and scope caveats.

**There are no remaining input-format or context blockers to running these231
public JevBench items.** The full run is complete, and accuracy favors GSO. Do not
present the public set as untouched held-out data after using it to diagnose
implementation gaps. No official sealed score/rank is available. A fresh GSO GPU
rerun remains separately gated by the GPU hold, not a blocker to this audited
historical accuracy comparison.

`RuntimeReady=false` remains appropriate for broader image/service/quality and
platform gates. This benchmark does not require further MoJev optimization work
before moving to the next model as directed.

## Evidence

`/workspace/tmp/mojev-jevbench140-parity-20260926` contains the exact server binary
and sources, original/resume runners and manifests,231 per-item records and raw
responses, full upstream summaries, per-item comparison CSV, tier/type/family
CSV, hash/audit outputs and preserved interrupted logs. The original200 records
are retained separately. Compatibility preflight evidence is under
`/workspace/tmp/mojev-systemone-parity-20260926`.

The downloadable evidence bundle omits model weights, executables and cloned
repositories. Public task material remains under the included JevBench licence.
Historical GSO evidence remains in its original repository, unchanged.
