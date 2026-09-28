## Best native JEV equivalent: Decider

Decider is the best JEV-like model among the native ports tested on this host. It reached **75.40%** accuracy on 378 untouched validation originals, compared with **62.96%** for OpenJEV, and was better on every task. Its normalized candidate probabilities also produced lower held-out NLL, Brier score and expected calibration error (ECE).

This is a new bake-off over existing non-test data. It does not reopen the completed 1,440-row Jevlike final evaluation, and none of those spent test rows was read or used for model selection.

## Frozen cohorts

The [policy](jev-port-bakeoff-policy.md) and common adapter were committed before scoring. `scripts/jev-port-bakeoff.ts` verified the prepared dataset manifest, validation partition, provenance and the prior 60-original development selection, then wrote two disjoint cohorts:

| Cohort | Originals | Requests | Purpose |
|---|---:|---:|---|
| Screening | 40 | 49 | Eight originals per task, plus order and evidence controls |
| Finalist | 378 | 804 | Every remaining untouched validation original, reverse-order copies, and bounded evidence controls |

The cohort hashes are:

| Artifact | SHA-256 |
|---|---|
| Selection | `1fa7c990d400afd7e24828faa2fccb62f23d6462b4f9320e30c098e6ab36859d` |
| Screening requests | `bd10c5959045d97d252fc797a5e9f9ffee4277a6e0d28a991b6e5cf55d950e1d` |
| Finalist requests | `5da27af532ef042b97b5d535864835701c90de43c003da1a10ae6897faaaa6f5` |
| Decider finalist results | `8240f660ad73c109ba3dc2b27d0455b88a6b31554d43c53fb9d78deb3cd55206` |
| OpenJEV finalist results | `69fa5b459ac181db38575b41a0f0785823c0e68c1f01eb09f95a487018b29620` |

Strict validation confirmed zero overlap between screening, finalist and the 60 spent development originals. Every scored row retained the original candidate text, stable candidate IDs and gold identity.

## Screening

Nimble failed the predeclared host resource gate before outcomes were inspected: its published native path peaks near 59 GiB RSS and takes about 234 seconds per field, exceeding the 48 GiB and 60-second/request limits. Laya, Decider and OpenJEV completed all 49 screening requests without rejection.

| Model | Normal accuracy | Order changes | p50 | p95 | Peak RSS |
|---|---:|---:|---:|---:|---:|
| Decider | 32/40, **80.00%** | 1/5 | 9.58 s | 15.34 s | 4.92 GiB |
| OpenJEV | 20/40, **50.00%** | 0/5 | 13.44 s | 25.32 s | 4.92 GiB |
| Laya | 18/40, **45.00%** | 3/5 | 2.24 s | 3.83 s | 2.21 GiB |
| Nimble | Not admitted | -- | -- | -- | Published 59 GiB |

The frozen rule promoted Decider and OpenJEV. Laya was faster, but its quality and order stability were weaker in this cohort.

## Finalist quality

Both finalists completed all 804 requests with no rejection. The normal cohort contains 378 originals; all 378 also have a reverse-order copy. MultiNLI and CLINC contribute 24 shuffled-evidence and 24 no-evidence controls in total.

| Model | Normal accuracy | Random expectation | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| Decider | **285/378, 75.40%** | 23.09% | **0.609** | **0.330** | **0.062** |
| OpenJEV | 238/378, 62.96% | 23.09% | 1.029 | 0.552 | 0.109 |

The probability metrics normalize each row's stored candidate scores within the offered set. They are supplemental held-out reporting: no temperature or threshold was fitted on screening or finalist data. `jevcompare -report-results ... -report-output ...` verifies the selection, request and result identities before computing accuracy, NLL, summed multiclass Brier score, ten-bin ECE, reliability bins and risk/coverage points.

Decider led on every task:

| Task | Examples | Decider | OpenJEV |
|---|---:|---:|---:|
| ARC Challenge | 26 | **73.08%** | 50.00% |
| ARC Easy | 44 | **86.36%** | 61.36% |
| CLINC OOS `plus` | 93 | **90.32%** | 89.25% |
| CommonsenseQA | 101 | **57.43%** | 47.52% |
| MultiNLI | 114 | **75.44%** | 58.77% |

CLINC is the prepared labelled eight-choice task, including OOS. It is not full 151-intent deployment accuracy.

## Order and evidence controls

OpenJEV was perfectly invariant to candidate order: **0/378** answers changed, and reverse-order accuracy remained 62.96%. Decider changed **48/378** answers (12.70%); reverse-order accuracy rose slightly to 76.46%. The stable-ID comparison means these are actual decision changes, not moved option positions.

| Model | Shuffled evidence | No evidence | Changed vs normal |
|---|---:|---:|---:|
| Decider | 1/24, 4.17% | 2/24, 8.33% | 20/24 shuffled; 19/24 absent |
| OpenJEV | 2/24, 8.33% | 3/24, 12.50% | 21/24 shuffled; 21/24 absent |

Both models depend strongly on evidence for the bounded MultiNLI/CLINC controls. OpenJEV has the cleaner order contract. Decider's 12.70% order-change rate is a real limitation, but the predeclared decision rule selects normal accuracy first; its 12.43-point accuracy lead is decisive.

## Runtime and memory

The frozen runs used `GOMAXPROCS=2 nice -n 10` on the Intel i7-12700 host. Timings include per-row scoring and synchronous result persistence. Model loading sits outside recorded per-row decision time.

| Model | Finalist wall time | p50 per request | p95 per request | Peak RSS |
|---|---:|---:|---:|---:|
| Decider | **2 h 05 m 25 s** | **9.69 s** | **15.68 s** | 4.98 GiB |
| OpenJEV | 3 h 51 m 34 s | 15.68 s | 26.26 s | 4.97 GiB |

Decider was about 1.84 times faster by wall time on the same 804-request cohort.

## Profile-driven SIMD review

Separate post-evaluation CPU profiles used the frozen 49-request screening cohort. They cannot change the winner or the finalist records. Baseline profiles attributed **79.4%** of Decider CPU samples and **82.0%** of OpenJEV samples to existing SIMD dense GEMV (`dotRowsx8Asm` and `dotRowsx4Asm`). The largest remaining scalar function was the Qwen3.5 gated-delta state-row update at 9.06% and 8.13% respectively.

Two implementations were tested and rejected:

* A full `VecScale`/`Sdot`/`VecScaleAdd` rewrite made the isolated 128-wide row kernel about 4.6 times faster, but changed one of 49 Decider selections and moved OpenJEV logits by as much as 0.1028. It failed numerical parity.
* A decay-only `VecScale` retained exact scores and selected IDs for both released models. Its isolated row kernel improved by about 10--15%, but OpenJEV's whole screening run regressed from 12m49s to 13m05s because the added vector pass did not reduce the scalar reductions or state update.

Neither change is shipped. The baseline arithmetic remains in the tree. The retained [profile summaries](../performance/jev-port-bakeoff-20260921/) record the measured bottlenecks and rejected variants; a useful next optimisation would need a fused assembly kernel that preserves the reference accumulation order or a separately approved tolerance contract.

## Decision and limits

Decider is the best current native JEV equivalent for this prepared mixed-task choice workload. It has the strongest accuracy, probability diagnostics and runtime among admitted candidates. OpenJEV remains preferable when exact candidate-order invariance is mandatory and the 12.43-point accuracy loss is acceptable.

The conclusion is bounded to the existing prepared tasks and released native checkpoints. It does not establish broad human preference judgement, production calibration, full-intent CLINC accuracy, vision support or arbitrary-schema behaviour. Simple-JEV was excluded under the frozen policy because its upstream repository lacked a root licence at selection time. Upstream added Apache-2.0 in `b02aa81c` after this evaluation; it did not participate in these cohorts.

Further model or prompt changes require another experiment and another untouched cohort. The spent 1,440-row Jevlike final set and these screening/finalist cohorts cannot become tuning data for this comparison.

## Reproduction

Generate the frozen cohorts before any scoring:

```bash
bun scripts/jev-port-bakeoff.ts

go build -o /tmp/jevcompare ./cmd/jevcompare
```

Score a candidate:

```bash
GOMAXPROCS=2 nice -n 10 /tmp/jevcompare \
  -arm decider \
  -study checkpoints/jev-port-bakeoff/v1 \
  -cohort finalist \
  -model /path/to/decider-0.8b \
  -output checkpoints/jev-port-bakeoff/v1/decider-finalist.jsonl
```

Generate the strict probability report without loading a model:

```bash
/tmp/jevcompare \
  -study checkpoints/jev-port-bakeoff/v1 \
  -cohort finalist \
  -report-results checkpoints/jev-port-bakeoff/v1/decider-finalist.jsonl \
  -report-output /tmp/decider-finalist-metrics.json
```

Add `-cpuprofile /tmp/decider.pprof` to a scoring run for a Go CPU profile. `-resume` validates every existing output row against the frozen request prefix and model identity before appending missing rows.
