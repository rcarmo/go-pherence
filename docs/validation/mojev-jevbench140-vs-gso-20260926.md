# MoJev on JevBench v1.4.0, compared with GSO

## Result

The current native MoJev text stack scored **103/231 (44.59%)** on the JevBench
v1.4.0 public set. The audited historical Go System One result is **196/231
(84.85%)**. All231 items were attempted once; failures count as wrong answers.

| Public tier | Items | MoJev correct | MoJev strict-valid | Historical GSO correct |
|---|---:|---:|---:|---:|
| Easy | 48 | 42 (87.50%) | 48 | 48 (100%) |
| Original | 72 | 49 (68.06%) | 72 | 71 (98.61%) |
| Hard | 111 | 12 (10.81%) | 21 | 77 (69.37%) |
| **All** | **231** | **103 (44.59%)** | **141 (61.04%)** | **196 (84.85%)** |

GSO returned231 strict-valid responses. MoJev returned141 strict-valid responses
and90 explicit HTTP422 refusals, all in the hard tier:

- **55:** a text branch exceeds the current512-token SIMD capacity.
- **35:** the state is a JSON object, unsupported by the current text-only parser.

No state was stringified, no prompt was rewritten or truncated, no failure was
retried, and no task was silently omitted. These results describe the current
native stack and interface, not intrinsic base-model capability alone.

## Same successful-input subset

Among the141 items producing valid MoJev outputs, MoJev scored **103/141
(73.05%)**, versus GSO **136/141 (96.45%)** on those exact same task IDs.

| Paired outcome | Items |
|---|---:|
| Both correct | 100 |
| Only MoJev correct | 3 |
| Only GSO correct | 36 |
| Both wrong | 2 |

This subset excludes90 hard-tier items and is biased toward easier inputs; it is
not a replacement headline score. It nevertheless shows that unsupported inputs
are not the whole gap: historical GSO also performs substantially better on the
inputs this MoJev implementation answered.

## Versions and protocol

**MoJev, fresh run:**

- Native go-pherence revision `4bbd706f`, all cumulative CPU optimizations retained.
- Approved0.8B checkpoint revision
  `0c8695b6252f4205907433d4e196a94f032e60c3`.
- Config, safetensors, tokenizer and tokenizer-config SHA-256 checks before load;
  safetensors SHA-256
  `eae27bf03e0e44501316cafab2406b1505732ddf4b836d19fbb8deb642f55f50`.
- F32 SIMD, capacity512, `GOMAXPROCS=6`, NVIDIA disabled. Go1.26.3/Linux amd64,
  reported Intel i7-12700. One checkpoint process; no concurrent model process.
- Temporary loopback-only HTTP wrapper, ephemeral port, calling existing
  `DecodeTextRequest` and `ScoreTextContext(...,4096,4096)` unchanged. It only
  maps owned answers/usage to the public wire shape and errors to422. It is an
  evaluation transport, not production HTTP-service qualification.

**GSO, historical run, not rerun here:**

- Standalone service revision `b18ee0d4748bac436999aa72c000e06406c3cce6`.
- Gemma4-12B IT `UD-Q4_K_XL`, NVIDIA RTX3060 12GB.
- Published evidence in standalone repository
  `docs/benchmarks/data/jevbench-v140-public-20260923/`, report commit
  `a73f987be3abf0d84fb51b489ccb007f2d84fa07`.
- Original231 requests/responses and aggregates were audited again offline with
  the pinned upstream scorer. They reproduce196 correct and zero failures.
- Original collection waited for GPU temperature at or below55°C before each
  request; waiting was excluded from decision latency. No retries/warmup/tuning.

The existing GPU hold was not treated as lifted by the comparison request. No
GPU kernel or existing service was started or restarted. A fresh GPU GSO run
remains pending explicit clearance; this report is a fresh MoJev run compared
with audited historical GSO evidence, not a contemporaneous head-to-head run.

**Benchmark:** `fstandhartinger/jevbench` revision
`2fa63fa3226cb369795525ed011800f57dcbd894`, v1.4.0 public datasets:
48 easy,72 original,111 hard;139 Choice,74 Noul,18 Score tasks.

The unmodified pinned Python `TypeSafeAdapter`, `Runner`, `score_task` and
`summarize` were used. Python is necessary here to reuse the actual upstream
scorer, not a locally rewritten scoring approximation. No paid API was called;
route/compute costs remain unknown/null rather than a fabricated zero price.

Requests were serial, with no warmup, retries, truncation, prompt tuning or
criterion/label changes. Loading was excluded from request latency. The runner
retained upstream stop rules;422 input refusals count as wrong, not infrastructure
outages. Each record was flushed/fsynced with raw request/response hashes.

All231 MoJev requests were compared against archived GSO requests and found
identical **except the model ID**. Returned distributions, predictions, raw
hashes and correctness were independently rechecked using the pinned scorer.
MoJev outputs needed no probability renormalization.

This is a **public-set observation**, not JevBench's official sealed score, rank
or cost-qualified result. Those values remain null. The public set is now an
observed development benchmark and must not later be called untouched held-out
data after changes are informed by it.

## Latency and memory

| Scope | MoJev CPU p50 / p95 | Historical GSO GPU p50 / p95 |
|---|---:|---:|
| All231 attempts | 392 / 1,639 ms | 374 / 15,751 ms |
| Same141 MoJev-valid items | 534 / 1,990 ms | 136 / 927 ms |

**Do not interpret the all-item tail difference as a MoJev speedup:** its quick
refusals replace90 hard requests, including long inputs GSO actually executed.
Even the common-subset comparison uses different model sizes, precision,
CPU/GPU execution and collection dates. It reports observed stack latency, not
controlled intrinsic model efficiency. One attempt per item also does not
measure repeated per-task latency distributions.

MoJev load/preparation took about4.02s; evaluation server total wall time was
1m53.46s. Peak process RSS was4,990,336KiB (about4.76GiB), zero swaps. Loaded
post-GC heap was3,181,134,064 bytes, final3,181,646,392 bytes. This includes the
local transport/tokenizer and is not an hours-long retention claim. No fresh GSO
memory measurement is available in this run, so historical microbenchmark RSS
is not mixed into this quality-suite comparison.

## Interpretation and next work

GSO's audited historical result is better on both full-suite outcomes and the
common successful-input subset. Fixing JSON-state compatibility and long-context
admission would remove important implementation limitations, but it would not
by itself close the supported-subset accuracy gap.

Do not tune prompts, labels, thresholds or scoring on this run and then present
a rerun as fresh held-out qualification. Any state-serialization support or
context expansion needs its own explicit contract, parity, ownership, memory and
cancellation checks. Convolution/GEMM optimizations preserve outputs and are not
responsible for these quality differences. Existing optimizations remain adopted.

A delegated review of the result interpretation found no misleading headline
provided the current-stack, hard-tier-selection and historical CPU/GPU caveats
remain explicit. A filesystem-based adapter audit timed out; the actual pinned
adapter, runner and scorer were then inspected locally and verified by the
request/hash/scoring audit. `RuntimeReady=false` remains unchanged.

## Evidence

`/workspace/tmp/mojev-jevbench140-20260926` contains the detached pinned suite,
temporary server and runner source, binary/hash,231 durable records and raw
responses, summaries, per-item CSV, source/task-hash comparison, host samples and
server time/RSS log. Original GSO evidence and frozen runs were not modified.
The owned evaluation server was shut down after completion.

Re-audit without inference:

```sh
python3 /workspace/projects/go-system-one/docs/benchmarks/data/jevbench-v140-public-20260923/audit.py \
  --suite /workspace/tmp/mojev-jevbench140-20260926/suite
python3 /workspace/tmp/mojev-jevbench140-20260926/compare.py
```

The downloadable bundle includes raw public evidence under the included
JevBench licence, runner/server sources and reports; model weights, executables
and the cloned repository are omitted. The CSV lists every attempt, including
refusals, expected/predicted labels and historical GSO outcomes.
