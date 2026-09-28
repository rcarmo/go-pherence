# MoJev model package retired

`model/mojev` was removed from the current Go tree on 27 September 2026 at the owner's direction. The bounded JevBench public231 evaluation recorded 137/231 (59.31%) for MoJev versus an audited historical Go System One result of 196/231 (84.85%). The two systems used different execution paths; those figures describe the recorded evaluation, not a controlled model-only comparison. The owner judged MoJev insufficient for continued work.

Historical validation reports, oracle scripts and GPU diagnostic sources remain in the repository for audit. Their Go commands and `model/mojev/testdata` references describe the removed package and are no longer runnable from this checkout. No MoJev inference API or model files ship from `model/` now. The removal does not change the underlying Qwen3.5 runtime or the separate Needle 3 work.

The released MoJev model had a documented cross-question isolation leak. An experimental native F32 text path used separate branch histories, but it did not qualify model quality or production readiness. The [historical roadmap](mojev-inference-roadmap.md) and [encoder contract](mojev-hybrid-encoder-contract.md) are retained as records of that investigation, not active implementation plans. No further MoJev work is scheduled.
