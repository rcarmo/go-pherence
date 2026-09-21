# Native JEV-equivalent bake-off CPU profiles

These summaries record the separate post-evaluation CPU profiles described in the [bake-off report](../../experiments/jev-port-bakeoff-report-20260921.md). They use the frozen 49-request screening cohort with `GOMAXPROCS=2 nice -n 10`.

The baseline profiles show that existing dense GEMV assembly accounts for 79.4% of Decider CPU samples and 82.0% of OpenJEV samples. The Qwen3.5 gated-delta state-row update is the largest remaining scalar function at 9.06% and 8.13%.

Two candidate rewrites were rejected:

* the aggressive SIMD rewrite changed one Decider selection and moved OpenJEV logits by up to 0.1028;
* the decay-only SIMD rewrite preserved exact released outputs but regressed OpenJEV wall time from 12m49s to 13m05s.

The tree retains the baseline arithmetic. The `.txt` files are `go tool pprof -top` summaries; [the Decider Sankey](decider-baseline-sankey.svg) is generated from the same baseline profile.
