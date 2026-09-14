# Source-faithful eight-row GEMV kernel — 14 September 2026

After the convolution offset hoist, `dotRowsx4Asm` remained 50.55% of sampled CPU. The retained amd64 experiment processes eight consecutive output rows per call so one activation load feeds eight weight rows. Each output row keeps the existing x4 kernel's two vector accumulators, ascending loop order and horizontal reduction exactly. Admission is limited to measured convolution reductions divisible by 16 and at most 2304; larger projection widths, other reductions and non-amd64 architectures use two existing x4 calls. The public checked `GemvRows` contract is unchanged.

## Exactness and direct kernel screen

Model-free tests compare every x8 output bit against two x4 calls at generic/tail widths and the real convolution K values 288, 576, 1152 and 2304. They also compare complete `GemvRows` output bits for row tails, model-sized shapes and the 4096/5120 large-width fallback. All pass.

Three-sample direct-kernel medians:

| K | Two x4 calls | One x8 call | Speedup |
|---:|---:|---:|---:|
| 288 | 70.30 ns | 50.14 ns | 1.402× |
| 576 | 143.9 ns | 93.62 ns | 1.537× |
| 1152 | 296.0 ns | 180.7 ns | 1.638× |
| 2304 | 605.0 ns | 530.6 ns | 1.140× |

These are cache-resident microbenchmarks, not pipeline timings.

## Pinned five-second trained trunk

Parent/candidate binaries ran A/B/A/B on the same resident public Fbank. All 161,280 F32 outputs had identical SHA-256 `dc9121878f784e986fe6cb87b0f4c6a93293ec8c7152311295b0655588ee2a6d`.

| Run | Parent | Candidate |
|---|---:|---:|
| 1 | 751.780 ms | 688.111 ms |
| 2 | 749.721 ms | 693.893 ms |
| Mean | **750.750 ms** | **691.002 ms** |

The full-trunk gain is 1.0865× (7.959%). Parent RSS samples were 93,416–101,472 KiB and candidate samples 109,580–125,868 KiB; all reported zero swaps. Separate-process RSS includes model loading and is not allocation attribution.

## Canonical cumulative SDM result

The canonical overlapped 60-second AMI SDM nested result remained byte-identical at `b3c8456fcececc7199d79a6a846b76239dd87fd165e6a83532784ff8f0f62798`.

| Path | Pipeline | Process wall | Max RSS | Package RAPL |
|---|---:|---:|---:|---:|
| Sequential retained baseline | 106.201 s | 106.62 s | 152,776 KiB | 1,491.641 J |
| Overlap + offset hoist | 78.710 s | 78.90 s | 156,668 KiB | 1,165.982 J |
| Overlap + offset hoist + x8 | **73.485 s** | **73.68 s** | 158,052 KiB | **1,093.290 J** |

Relative to the preceding retained path, x8 improves latency 6.638% (1.0711×) and package-counter energy 6.234%. Relative to the original sequential path, the cumulative latency reduction is 30.805% (1.4452×). Process swaps were zero; package temperature was 42→47°C.

RAPL is a host package-counter delta around the command, not process-attributed laboratory energy. The baseline was not rerun. This qualifies only the exact-output CPU slice; strict Community intermediates/default ties, broad DER/JER, HTTP, ARM execution and complete-plan targets remain unqualified.
