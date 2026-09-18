# AMI single-distant-microphone pilot — 13 September 2026

## Result

The pinned PyAnnote Community-1 source pipeline and Go implementation were run serially on the same 60-second `ES2004a` excerpt using the official `Array1-01` **SDM** signal. This is the standard single-distant-microphone condition, unlike the initial summed-headset (`ihm-mix`) pilot. Inputs, timeline and references are hash-pinned in `../ami-pilot-contract-20260913/sdm-source.json`.

Automatic speaker count produced three clusters, 17 full turns, 13 exclusive turns and 77 diagnostic reconstruction ties.

Absolute AMI quality with overlap included:

| Output | Collar | DER | JER |
|---|---:|---:|---:|
| Full turns | 0 s | 51.3122% | 65.5871% |
| Full turns | 0.25 s | 48.4980% | 65.0998% |
| Exclusive turns | 0 s | 56.4677% | 70.3013% |
| Exclusive turns | 0.25 s | 53.4649% | 70.0701% |

Relative to the same interval from the headset mix, SDM improves full DER by **1.4833 percentage points** and full JER by **5.3774 points** at collar 0; ties fall from 217 to 77 and automatic clusters increase from two to three. The SDM condition is retained as a real incremental improvement and the preferred fixed condition for subsequent AMI work. It remains far from qualifying quality.

## Go/source parity

- Segmentation: 90,117 / 90,117 values exact.
- Embeddings: 12,701 / 39,168 exact; max absolute delta **2.7418137e-6**.
- Full/exclusive turns: exact under one three-label permutation.
- DER/JER delta: exactly 0 percentage points at collars 0 and 0.25.

The source pipeline has the same three-cluster output and quality. SDM therefore improves the evaluated signal condition without exposing a Go/source graph regression.

## Runtime/resources

| Runtime | Wall | Max RSS | Process swaps | Package RAPL | Mean package power |
|---|---:|---:|---:|---:|---:|
| Go | 106.62 s | 152,776 KiB | 0 | 1,491.641 J | 13.969 W |
| Pinned source | 132.07 s | 775,820 KiB | 0 | 1,977.876 J | 14.959 W |

This is not a balanced speed claim: source startup/hook writes and Go diagnostic serialization differ. The source-parity and within-Go condition comparisons are the valid conclusions.

## Cumulative decision

`qualified:false`. Small gains are retained and should be compounded rather than discarded. Future kernel, batching and postprocess changes must be measured both independently and cumulatively on this exact SDM fixture, with source parity and absolute DER/JER rechecked after every stack. No accumulation of performance gains can compensate for failed quality or strict tie gates; both dimensions must pass separately.
