# Float32 finiteness predicate — 14 September 2026

Post-x8 profiling attributed 5.16 seconds flat to `math.IsInf` inside Community-1 finiteness scans. Those scans widened each float32 and called both `math.IsNaN` and `math.IsInf`. The retained predicate instead checks whether the float32 exponent bits are all ones. This rejects every NaN payload and both infinities while accepting signed zero, subnormals and all finite normals.

No scan, cancellation boundary or failure point is removed. The same predicate is used for BatchNorm coefficients and block tensors; arithmetic and output bytes are unchanged.

## Model-free verification

Full Community-1 tests and vet pass. Explicit classification covers +0, -0, positive/negative smallest subnormal, ±max finite, ordinary finite values, ±Inf and multiple positive/negative NaN payloads.

A focused three-sample benchmark scans 65,536 finite values:

| Predicate | Median |
|---|---:|
| Widen + `IsNaN` + `IsInf` | 79.925 µs |
| Float32 exponent bits | 34.731 µs |

The isolated predicate speedup is 2.301×. This is not a pipeline timing.

## Canonical cumulative SDM result

The canonical overlapped 60-second AMI SDM nested result remained byte-identical at `b3c8456fcececc7199d79a6a846b76239dd87fd165e6a83532784ff8f0f62798`.

| Path | Pipeline | Process wall | Max RSS | Package RAPL |
|---|---:|---:|---:|---:|
| Sequential retained baseline | 106.201 s | 106.62 s | 152,776 KiB | 1,491.641 J |
| Previous retained CPU path | 71.847 s | 72.04 s | 157,152 KiB | 1,074.040 J |
| Above + exponent-bit predicate | **68.338 s** | **68.53 s** | 159,288 KiB | **1,045.312 J** |

Relative to the preceding path, this removes 4.884% latency (1.0513×) and 2.675% package-counter energy. Relative to the original sequential path, cumulative latency falls 35.652% (1.5541×) and package-counter energy falls 29.922%. Process swaps were zero; package temperature was 44→48°C.

RAPL is a host package-counter delta around the command, not process-attributed laboratory energy. The baseline was not rerun. This qualifies only the exact-output CPU slice; strict Community intermediates/default ties, broad DER/JER, HTTP, ARM trained execution and complete-plan targets remain unqualified.
