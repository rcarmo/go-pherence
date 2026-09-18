# Trained combined AMI HTTP pilot — 13 September 2026

## Result

The hash-pinned 60-second `ES2004a` AMI excerpt was submitted to a real temporary `speechjobserve` instance using the trained CPU seven-stage profile:

`decode → asr-windows → transcript → vtt → diarization → speaker-transcript → speaker-vtt`

The job failed closed at speaker publication, as required. Plain transcript/VTT remained public and downloadable; internal diagnostic diarization was retained at checkpoint 5. No `speaker-transcript` or `speaker-vtt` artifact was published because the retained result contains 217 ambiguous frames under explicit `LowestIndexTies` diagnostics.

The committed `checkpoints/` copies were read directly from the private store after shutdown for offline scoring. Their inclusion here is diagnostic evidence, not proof that the HTTP API exposed raw diarization or speaker labels.

## Transcript and private diagnostic attribution

- 960,000 exact canonical samples (60 s).
- 15 plain transcript cues.
- 128 timed word records, normalizing to 127 scored lexical tokens; one output record is punctuation-only.
- 179 independently transcribed AMI reference words across four speakers.
- Ordinary WER: **43.5754%** — 22 substitutions, 54 deletions, 2 insertions.
- Private diagnostic cpSA-WER: **63.1285%** — 108 speaker-stream edit errors plus 5 unlabelled-token errors.
- Community output: 2 clusters for 4 reference speakers; 217 ambiguous frames.

The cpSA-WER scorer performs optimal speaker-stream permutation and does not infer reference words or speakers from system output. Here it attributes the retained plain words in memory from the diagnostic exclusive turns; it does not serialize a speaker-labelled transcript. This result is `qualified:false`.

## Runtime and resources

- HTTP job run: **109.195 s**; process wall including startup/load/shutdown: **110.13 s**.
- Maximum process RSS: **469,980 KiB**.
- Process swaps: **0**.
- Host package RAPL delta: **1,498.793 J**, or **13.608 W** mean over the 110.14-second bounded command.
- Raw `psys` delta: 23.212 J, retained without interpreting platform-specific domain semantics.
- Endpoint package temperature: 40→47 °C. This is not a continuous thermal trace.

The whole job is 0.550× realtime and far below the planned ≥5× complete-pipeline target. It also has poor lexical and speaker quality, so performance and quality both fail on this pilot.

## Scope

This is one overlap-heavy AMI evaluation excerpt with Whisper Tiny and Community-1. It is the first independent timed-word/multi-speaker combined-service diagnostic, not broad-corpus qualification. Tiny is not the intended release-quality multilingual model, strict Community tie handling rejects this sample, and no diagnostic speaker labels are promoted to users.
