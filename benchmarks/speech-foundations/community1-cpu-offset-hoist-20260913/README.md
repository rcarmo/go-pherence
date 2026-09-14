# Community-1 CPU convolution offset hoist — 13 September 2026

The active source-faithful `WeSpeakerBlockSIMD` convolution previously recomputed each spatial coordinate inside the input-channel loop. Spatial coordinates depend only on output position and kernel tap. This change computes the at-most-nine source offsets once per output position, then fills the same channel-major/tap-major patch with explicit positive-zero padding.

Floating-point arithmetic, GEMV order, weight layout, output scatter, cancellation checks and public modes are unchanged. This is not the faster experimental tiled-GEMM mode, whose distinct reduction order remains unqualified by the strict trained comparisons.

## Correctness

Targeted model-free Community-1 convolution, full-depth oracle and cancellation tests passed. On the canonical overlapped 60-second AMI SDM fixture, the nested result stayed byte-identical to the retained sequential/overlap baseline:

```text
b3c8456fcececc7199d79a6a846b76239dd87fd165e6a83532784ff8f0f62798
```

That covers 51 windows, segmentation, masks, embeddings/support, clustering, three speakers, 17 full turns, 13 exclusive turns and 77 diagnostic ties. The outer JSON differs only in measured elapsed metadata.

## Resident five-second trunk screen

Parent/current binaries were built from the same revision, with only the offset-hoist working-tree change in the candidate. Runs alternated A/B/A/B against the pinned five-second Fbank. All 161,280 F32 outputs had identical SHA-256:

```text
dc9121878f784e986fe6cb87b0f4c6a93293ec8c7152311295b0655588ee2a6d
```

| Run | Parent | Candidate |
|---|---:|---:|
| 1 | 920.469 ms | 736.099 ms |
| 2 | 917.980 ms | 748.228 ms |
| Mean | **919.225 ms** | **742.163 ms** |

The narrow speedup is **1.2386×** (19.262%). Parent RSS samples were 94,872–116,544 KiB and candidate samples were 125,952–125,976 KiB; all runs reported zero swaps. The RSS samples include separate test-process/model loading effects and are not a controlled allocation attribution.

## Canonical cumulative SDM result

| Path | Pipeline | Process wall | Max RSS | Package RAPL |
|---|---:|---:|---:|---:|
| Sequential retained baseline | 106.201 s | 106.62 s | 152,776 KiB | 1,491.641 J |
| Overlap retained baseline | 96.755 s | 97.17 s | 156,108 KiB | 1,414.020 J |
| Overlap + offset hoist | **78.710 s** | **78.90 s** | 156,668 KiB | **1,165.982 J** |

Relative to overlap alone, the retained candidate is **1.2293×** faster (18.650%), uses 17.541% less package-counter energy and adds 560 KiB measured RSS (0.359%). Relative to the original sequential path, the cumulative reduction is 25.886% (1.3493×). Process swaps were zero. Package temperature was 41→47°C in the candidate run.

RAPL is a host package-counter delta around the command, not process-attributed laboratory energy. The retained baseline was not rerun in this window. This result qualifies only this source-faithful CPU slice; it does not resolve strict Community intermediate/default-tie failures, broad DER/JER quality, HTTP latency, ARM support or complete-plan targets.
