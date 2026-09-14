# Trained combined CPU service diagnostic — 13 September 2026

## Result

A real temporary loopback `speechjobserve` instance loaded hash-pinned Whisper Tiny and Community-1 CPU models and exercised the actual seven-stage HTTP profile:

`decode → asr-windows → transcript → vtt → diarization → speaker-transcript → speaker-vtt`

The 30-second public pyannote tutorial sample reached five durable checkpoints. Plain transcript/VTT and internal diarization were retained, but speaker publication failed closed because the diagnostic `LowestIndexTies` result contains 84 ambiguous reconstruction frames. This is expected safety behavior: diagnostic tie resolution is not promoted into speaker-labelled output.

The same loaded server then processed the pinned 11-second JFK sample three times. Every run completed all seven stages. The four public artifacts were byte-identical across repeats. Each transcript contains 22 checked words; 21 receive the single Community speaker and one remains unlabelled because it has no positive exclusive-turn overlap.

## Timings and resources

- Server startup ready, including hash checks and loading both trained model families: **0.575814 s**.
- Three warm complete JFK HTTP jobs: **5.507839 s**, **5.534696 s**, **5.516508 s**; median **5.516508 s**.
- Whole test process: **63.80 s**, including one 30-second ambiguous-retention job, three JFK jobs, HTTP transfers, and shutdown.
- Maximum process RSS from `/usr/bin/time -v`: **470,988 KiB**.
- Process swaps: **0**. Host swap usage remained exactly `3,934,879,744` bytes; no new host swap was consumed.
- Available host memory rose from `29,272,064,000` to `29,377,228,800` bytes across the run.
- Thermal-zone snapshots: zone1 45→50 °C, zone2 46→44 °C. These endpoint snapshots are not continuous package-energy or throttling evidence.

## Scope

This is trained integrated-service, persistence, deterministic warm-run, memory and bounded thermal evidence on two short public samples. It is CPU-only and Linux/amd64-only; it does not qualify the Community Vulkan server profile. Tiny is not a representative multilingual release model, the two-speaker arm cannot publish labels under the existing ambiguity policy, and JFK is single-speaker English. The result therefore does not qualify broad multilingual WER, SA-WER, broad DER/JER, natural long-form recovery, the ≥5× complete-pipeline target, or energy efficiency.

The reusable transcript/alignment/scoring layers compile for Linux/ARM64. The trained server command does not: its Community construction/runtime types are deliberately guarded by `linux && amd64`. ARM remains a later port/qualification track requiring architecture-specific kernels, ownership checks and execution on real ARM hardware; this diagnostic neither weakens that boundary nor claims ARM support.

The opt-in gate is `make speech-job-trained-combined-check`; it requires explicit local asset paths and an authorised model/CPU window. No persistent service, deployment, private recording, or model download was used.
