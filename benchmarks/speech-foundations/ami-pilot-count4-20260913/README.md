# AMI explicit four-speaker ablation — 13 September 2026

## Question

Does supplying the known AMI reference speaker count fix the poor automatic-count result, while preserving Go/source parity?

Both the pinned PyAnnote source oracle and Go Community-1 pipeline were run on the same hash-pinned 60-second `ES2004a` excerpt with explicit `num_speakers=4`. The source used one Torch thread/MKLDNN off; Go used two threads/CGo and NVIDIA disabled. Runs were serial in one coordinated quiet window.

## Result

Both implementations select the existing forced K-means path and emit four clusters, 17 full turns and 13 exclusive turns. Go retains 60 ambiguous reconstruction frames under explicit lowest-index diagnostics.

Go/source parity remains strong:

- segmentation: 90,117 / 90,117 values exact;
- embeddings: max absolute delta 4.0531158e-6;
- full/exclusive turns: exact under one four-label permutation;
- DER/JER deltas: exactly 0 percentage points at collars 0 and 0.25.

Absolute AMI metrics remain poor:

| Output | Collar | DER | JER |
|---|---:|---:|---:|
| Full turns | 0 s | 52.8261% | 61.0112% |
| Full turns | 0.25 s | 50.9007% | 59.9406% |
| Exclusive turns | 0 s | 59.0110% | 67.5982% |
| Exclusive turns | 0.25 s | 56.9274% | 66.8797% |

Compared with automatic count, full JER improves from 70.96% to 61.01%, but full DER is essentially unchanged (52.80%→52.83%) and exclusive DER worsens (56.67%→59.01%). Known count is therefore not a quality fix and is not promoted as a default.

## Runtime/resources

| Runtime | Wall | Max RSS | Process swaps | Package RAPL |
|---|---:|---:|---:|---:|
| Go | 106.42 s | 153,260 KiB | 0 | 1,476.905 J |
| Pinned source | 133.26 s | 775,552 KiB | 0 | 1,978.492 J |

This is not a balanced performance claim: Python startup/hook writes and Go diagnostic serialization differ. The only acceptance assertion is source-output parity.

## Decision

`qualified:false`. Explicit count confirms that automatic count selection is not the sole quality blocker. The AMI headset mix may itself be a difficult/nonstandard diarization signal, but neither that possibility nor exact source parity makes these absolute metrics acceptable. Further corpus work should add standard distant-microphone AMI evaluation audio and more meetings, not tune to this one excerpt.
