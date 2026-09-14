# Trained Community-1 AMI pilot — 13 September 2026

## Result

The hash-pinned 60-second `ES2004a` AMI evaluation excerpt was run through the real trained Go Community-1 CPU pipeline with `GOMAXPROCS=2`, CGo and NVIDIA disabled. Source/model hashes and corpus preparation are documented in `../ami-pilot-contract-20260913/`.

The default `RejectAmbiguousTies` run failed closed at reconstruction frame 59. No result artifact was published. An explicit `LowestIndexTies` diagnostic run then completed, retaining **217 ambiguous frames**. It produced **2 clusters** for a four-speaker reference, so it is not a qualifying speaker result.

Absolute metrics over the fixed 0–60 s UEM, with overlap included:

| Output | Collar | DER | JER |
|---|---:|---:|---:|
| Full turns | 0 s | 52.7955% | 70.9645% |
| Full turns | 0.25 s | 49.9681% | 70.3948% |
| Exclusive turns | 0 s | 56.6691% | 71.5588% |
| Exclusive turns | 0.25 s | 53.5883% | 70.5322% |

These are poor absolute scores and reinforce the existing Community-1 qualification hold. They are not a reference-system delta and do not justify changing the fixed numerical or tie-policy gates.

## Runtime and resource evidence

| Mode | Result | Pipeline time | Process wall | Max RSS | Process swaps | Package RAPL | Mean package power |
|---|---|---:|---:|---:|---:|---:|---:|
| Strict | fail closed | n/a | 106.58 s | 165,348 KiB | 0 | 1,459.090 J | 13.670 W |
| Lowest-index diagnostic | complete | 106.510 s | 106.91 s | 160,668 KiB | 0 | 1,463.391 J | 13.673 W |

RAPL values are host package-counter deltas around each bounded command, not process-attributed laboratory measurements. The `psys` domain is retained raw because its platform-specific value (22.415/22.567 J) is not comparable to package energy here. Endpoint package temperatures were 40→47 °C for strict and 58→50 °C for diagnostic; they are not continuous thermal traces.

## Diagnostic geometry

- 960,000 canonical mono16k samples; 51 overlapping 10-second windows at 1-second steps.
- 60 admitted clustering rows.
- 2 output clusters.
- 22 full turns and 16 exclusive turns.
- 217 diagnostic reconstruction ties.
- CPU runtime was 1.775× the source duration, or 0.563× realtime throughput.

## Scope

This is one deliberately overlap-heavy AMI excerpt and remains `qualified:false`. It improves the evidence from a single tutorial clip to independently annotated meeting speech, but one excerpt is not broad-corpus qualification. Speaker-attributed WER still requires a successful seven-stage ASR+speaker result; strict policy currently prevents publication, and diagnostic labels are not promoted as valid speaker output.
