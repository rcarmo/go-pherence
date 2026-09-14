# AMI SDM overlapped CPU branches — 13 September 2026

## Change

`ExperimentalDiarization.RunPCMOverlapped` explicitly executes the independent segmentation and embedding-trunk branches concurrently for each immutable PCM window. Mask selection and mask-dependent pooling/projection still wait for segmentation and use the completed trunk. The existing sequential `RunPCM` remains unchanged and is still the default.

Both branches preserve their exact arithmetic and whole-window normalization. A branch error cancels the sibling, both workers are joined, and no partial result is returned. Observer notifications remain in the original public order: read, segmentation, masks, embedding.

## Correctness

Model-free checks cover real overlap, sibling cancellation, joined error preservation, nil results, immutable PCM, read schedule, observer order, and complete sequential/overlapped result parity. The full Community package and vet pass.

On the trained 60-second AMI SDM fixture, the canonical nested `Result` JSON is byte-identical between sequential and overlapped paths:

```text
b3c8456fcececc7199d79a6a846b76239dd87fd165e6a83532784ff8f0f62798
```

This identity covers 51 windows, segmentation, embeddings/support, clustering, 3 speakers, 17 full turns, 13 exclusive turns, and 77 diagnostic ties. The wrapper hash differs only because `ElapsedNanos` is measured metadata.

## Performance and energy

| Mode | Pipeline time | Process wall | Max RSS | Package RAPL | Mean package power |
|---|---:|---:|---:|---:|---:|
| Sequential retained baseline | 106.201 s | 106.62 s | 152,776 KiB | 1,491.641 J | 13.969 W |
| Overlapped | 96.755 s | 97.17 s | 156,108 KiB | 1,414.020 J | 14.530 W |

- Speedup: **1.0976×**.
- Latency reduction: **8.895%**.
- Package-energy reduction: **5.204%**.
- Process swaps: zero in both runs.
- Peak RSS increase: 3,332 KiB (2.18%).

RAPL is a host package-counter delta around each bounded command, not process-attributed laboratory measurement. Higher mean package power is expected from two active branches; shorter duration still lowers total package energy.

## Cumulative decision

Retain the explicit overlap path. It is a small, source-parity-preserving improvement that can compound with SincNet, recurrent, CNN and scheduling work. It is not promoted as the service default until wider corpus and contention tests pass. It does not improve DER/JER, resolve strict Community intermediates/default ties, or make the 0.620×-realtime Community stage meet the planned speaker/complete-pipeline targets.

`qualified:false` for the complete plan; `accepted:true` for this opt-in performance slice.
