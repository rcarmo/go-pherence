# Trusted internal ResNet chain — 14 September 2026

Every public `WeSpeakerBasicBlock.Forward` call validates arbitrary input finiteness. The owning full ResNet previously repeated that scan before all 16 internal blocks even though the stem and each preceding block return finite owned output. The retained change adds a private trusted entry used only by the unobserved full-model chain. Public block calls and every observed ResNet path keep the original scan, including detection of forbidden non-finite callback mutation before the next block.

All convolution, BatchNorm, residual, ReLU, output validation, cancellation and arithmetic remain unchanged. This removes only redundant internal input scans.

## Contract verification

Full model-free Community-1 tests and vet pass. New checks prove:

- trusted/unobserved and fully validated observed trunks are bit-identical in scalar and SIMD modes;
- public block NaN/Inf input rejection remains unchanged;
- if an observer violates its read-only contract by writing NaN, execution still fails before the next boundary.

## Pinned five-second trained trunk

Parent/current binaries ran the unobserved production trunk in A/B/A/B order. A separate current observed-path run retained the scan. All five outputs contained 161,280 F32 values with identical SHA-256 `dc9121878f784e986fe6cb87b0f4c6a93293ec8c7152311295b0655588ee2a6d`.

| Run | Parent | Trusted candidate |
|---|---:|---:|
| 1 | 695.229 ms | 685.439 ms |
| 2 | 695.416 ms | 687.733 ms |
| Mean | **695.322 ms** | **686.586 ms** |

The narrow gain is 1.0127× (1.257%). The observed candidate took 701.339 ms in one diagnostic sample. Parent RSS samples were 116,692–126,012 KiB and candidate samples 96,944–116,772 KiB; all reported zero swaps. Separate-process RSS is not controlled allocation attribution.

## Canonical cumulative SDM result

The canonical overlapped 60-second AMI SDM nested result remained byte-identical at `b3c8456fcececc7199d79a6a846b76239dd87fd165e6a83532784ff8f0f62798`.

| Path | Pipeline | Process wall | Max RSS | Package RAPL |
|---|---:|---:|---:|---:|
| Sequential retained baseline | 106.201 s | 106.62 s | 152,776 KiB | 1,491.641 J |
| Overlap + offset hoist + x8 | 73.485 s | 73.68 s | 158,052 KiB | 1,093.290 J |
| Above + trusted internal chain | **71.847 s** | **72.04 s** | 157,152 KiB | **1,074.040 J** |

Relative to the preceding retained path, this removes 2.229% latency (1.0228×) and 1.761% package-counter energy. Relative to the original sequential path, the cumulative latency reduction is 32.348% (1.4782×). Process swaps were zero; package temperature was 43→47°C.

RAPL is a host package-counter delta around the command, not process-attributed laboratory energy. The baseline was not rerun. This qualifies only the unobserved internal CPU chain; strict Community intermediates/default ties, broad DER/JER, HTTP, ARM trained execution and complete-plan targets remain unqualified.
