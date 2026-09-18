# Rejected amd64 dotRowsx4 FMA interleave — 14 September 2026

After the source-faithful convolution offset hoist, `dotRowsx4Asm` remained 50.55% of sampled CPU. A minimal experiment reordered independent row instructions in each 16-column loop: first-half FMAs for rows 0–3 were issued before second-half FMAs for rows 0–3. Each row retained the same accumulator, FMA order and horizontal reduction.

Exact `DotRowsx4`/`GemvRows` parity tests passed. The existing absolute microbenchmark also passed, but there was no retained parent measurement, so no gain was inferred from it.

A separate balanced parent/candidate screen built both binaries from the same revision and ran five 500 ms samples per shape in A/B/B/A process order. Process medians were averaged:

| K | Parent | Candidate | Change |
|---:|---:|---:|---:|
| 256 | 32.675 ns | 32.750 ns | −0.230% |
| 1024 | 132.000 ns | 132.250 ns | −0.189% |
| 4096 | 542.850 ns | 542.550 ns | +0.055% |

The result is noise-level with two slight regressions, not a stable gain. The assembly change was reverted and no trained/corpus run was justified. This evidence records a rejected optimization, not a retained implementation.
