# Fused 8-row × 16-token production audit — 2026-08-09

## Decision

Retain and promote the fused Q4_0x8/Q8_0x4 projection as a functionally valid CPU winner. It closes roughly one third of the remaining paired prompt gap, but does **not** reach the 89.405 tok/s target. The remaining blocker is native fused projection execution, not activation preparation or Go scheduling.

## Performance mechanism

The promoted path changes the arithmetic topology rather than rearranging the retained work:

- eight Q4_0 rows share each activation load and activation-sum correction;
- 16 tokens are accumulated per native call;
- Q4 nibbles use the unsigned correction identity `sum((q - 8) * a) = sum(q * a) - 8 * sum(a)`;
- F32 activations are quantised directly into Q8_0x4 without an intermediate Q8_0 buffer;
- row groups are scheduled dynamically across six workers, with the caller participating;
- weights are packed once while loading selected projection matrices, and canonical `Raw` storage is released rather than duplicated.

Embedding/head storage remains canonical and tied. Shared K/V matrices remain pointer aliases. `DequantRowTo` reconstructs a canonical Q4_0 row from replacement storage when required. Unsupported builds and CPUs retain canonical storage and the existing paths.

## Correctness and fallback gates

Passed:

- exhaustive fused tile/reference, supertile, row-tail, token-tail and dynamic-scheduling tests;
- replacement ownership and exact row dequantisation tests;
- deterministic finite replacement projection test;
- Q4_0 batch 1, 2 and 4 continue to wrap `ErrUnsupportedBatchProjection`;
- cgo and `CGO_ENABLED=0` focused suites;
- native C compile with `-Wall -Wextra -Werror`;
- real 124-token finite prefill;
- real one-step session, checkpoint restore and bit-identical restored logits;
- deterministic frozen fused 124+48 token trajectory;
- five alternating pinned baseline/candidate 124+48 requests.

Real-model evidence:

- `fused-8x16-real-prefill-functional-20260809.log`
- `fused-8x16-real-checkpoint-logits-20260809.log`
- `fused-8x16-vs-ea90f25b-paired-124x48-20260809.log`

## Clean projection gate

Pinned configuration: `taskset -c 0-5`, `GOMAXPROCS=6`, five samples.

| Complete projection path | Median ns/op | Change versus retained |
| --- | ---: | ---: |
| retained F32 production | 552,163 | — |
| fused direct-Q8 | 460,070 | **-16.68%** |

The one-time Q4 packing cost is excluded from request execution because loader ownership replaces canonical projection storage. Prepacked timing remains attribution only.

Source: `fused-8x16-clean-projection-20260809.log`.

## Paired 124+48 gate

Baseline is clean detached `ea90f25b`. Candidate and baseline alternated for five pairs under `taskset -c 0-5`, `GOMAXPROCS=6`.

| Phase | ea90f25b median | Fused median | Change |
| --- | ---: | ---: | ---: |
| prompt compute | 52.647171 tok/s | 64.799044 tok/s | **+23.08%** |
| decode evaluation | 9.646955 tok/s | 9.514437 tok/s | -1.37% |
| combined 124 prompt + 47 decode evaluations | 23.656570 tok/s | 24.950864 tok/s | **+5.47%** |

The fused prompt median is 72.48% of the explicit 89.405 tok/s target. Relative to the paired ea90f25b median, it closes 33.06% of the remaining target gap. It is 71.03% of the 91.229561 tok/s frozen llama.cpp prompt oracle used by the test.

## Remaining blocker

A pinned prefill profile attributes 81.08% of aggregate CPU samples to `runtime.cgocall` beneath `ProjectQ4_0x8Q8_0x4RowsVNNI`; the samples are summed across native worker calls. Activation preparation and Go scheduling are no longer dominant. The next material improvement therefore requires less native fused-kernel work — most plausibly a better register/spill schedule or ISA-specific dot-product topology — rather than another wrapper, packing, or traversal change.

Evidence:

- `fused-8x16-prefill-20260809.pprof`
- `fused-8x16-prefill-pprof-top-20260809.txt`
- `fused-8x16-prefill-profile-run-20260809.log`

The candidate is promoted because it is functionally valid and materially faster for the target prompt workload, while the unresolved 89.405 tok/s gap is recorded explicitly rather than claimed closed.
