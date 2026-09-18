# Trained AMI HTTP service with CPU branch overlap — 13 September 2026

## Result

The exact retained 60-second AMI headset-mix fixture was rerun through the trained temporary seven-stage HTTP service with only `profile.community.modes.overlap_branches:true` changed. The profile identity therefore changed intentionally and incompatible sequential checkpoints cannot resume under the overlapped profile.

The safety/output contract is unchanged:

- 960,000 exact canonical samples;
- 15 plain cues and 128 timed word records;
- 2 Community clusters and 217 diagnostic ties;
- 5 durable checkpoints;
- plain transcript/VTT retained;
- speaker publication refused (`speaker_publication:false`).

The plain transcript is byte-identical to the retained sequential run (`3d617350…fe28`). Diarization differs in its `stage_key`, as required by changed execution identity; after deleting only that key, canonical JSON is byte-identical (`d1316d6d…38fd`). No numerical/model/quality output changed.

## Cumulative performance

| Mode | HTTP job | Process wall | Max RSS | Package RAPL | Mean package power |
|---|---:|---:|---:|---:|---:|
| Sequential retained baseline | 109.195 s | 110.13 s | 469,980 KiB | 1,498.793 J | 13.589 W |
| Overlapped | 100.621 s | 101.54 s | 472,944 KiB | 1,484.541 J | 14.596 W |

- End-to-end speedup: **1.0852×**.
- End-to-end latency reduction: **7.852%**.
- Package-energy reduction: **0.951%**.
- Process swaps: zero in both runs.

The Community-only SDM measurement improved 8.895% latency and 5.204% energy. Complete HTTP gains are diluted by unchanged ASR, persistence and orchestration, but remain real and stackable. Higher mean package power is expected from concurrent CPU branches; shorter wall time still reduces total package energy slightly.

## Decision

Retain the explicit CPU overlap profile setting and its separate checkpoint identity. It is not the default and does not qualify the pipeline: the fixture remains 43.58% WER / 63.13% private diagnostic cpSA-WER, strict ties still reject speaker publication, and 0.596× realtime remains far below the ≥5× complete-pipeline target. Future improvements should be measured independently and cumulatively on fixed fixtures.
