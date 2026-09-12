# Community-1 public-corpus qualification contract

The current manifest has one diagnostic case and no qualifying corpus case. It cannot promote Community-1 or establish general diarization quality.

## Saved-result contract

[`community1-corpus-manifest.json`](community1-corpus-manifest.json) pins each canonical mono 16 kHz WAV, RTTM, Go result and source-reference result. It also fixes:

- dataset role: `diagnostic` or `qualifying`;
- licence and redistribution review state;
- language, split and URI;
- evaluated UEM intervals, collars and the maximum raw padded-turn extent;
- reconstruction tie policy;
- whether exclusive turns may overlap after configured gap filling;
- maximum absolute Go/reference DER and JER deltas in percentage points.

`score_community1_corpus.py` checks hashes, PCM geometry, RTTM extents, exact top-level result keys, identity fields used by the scorer and turn geometry before importing `pyannote.metrics`. It permits raw frame-centre turns through the manifest's pinned padded-tail extent; UEM still bounds scoring to source audio. It scores full and exclusive turns with overlap included. A diagnostic case can fail parity, but it cannot set `qualified=true`. A qualifying case requires verified redistribution terms.

The retained 30-second pyannote tutorial sample passes zero-delta DER and JER comparisons at 0 s and 0.25 s collars. Its manifest role is `diagnostic`: the public asset is retained locally, but its redistribution terms are not pinned. This does not count towards corpus coverage.

SA-WER is outside this scorer. The Community result has diarization turns but no word-level alignment. A future combined Whisper/speaker gate must pin the transcript normalisation, word timing, speaker assignment and reference transcript before reporting SA-WER.

## Candidate corpus matrix

No candidate below has been downloaded or run in this checkpoint.

| Corpus | Licence status checked on 12 September 2026 | Coverage | Admission work |
|---|---|---|---|
| AMI Meeting Corpus | CC BY 4.0 corpus/annotations on the University of Edinburgh licence page | English meetings, overlap, multi-speaker | Select public close-talk and array slices; pin WAV/RTTM/UEM and source reference |
| VoxConverse | README states CC BY 4.0 for research; original video owners retain copyright | English broadcast/interview speech, overlap, varied speakers | Review source-video redistribution limits; pin dev/test clips and RTTM |
| AliMeeting SLR119 | CC BY-SA 4.0 dataset on OpenSLR; recipe code is Apache 2.0 | Mandarin meetings, 2–4 speakers, overlap | Select bounded far-field clips; convert annotations to pinned RTTM/UEM |
| AISHELL-4 SLR111 | CC BY-SA 4.0 on OpenSLR | Mandarin meetings, 4–8 speakers, overlap | Select bounded test clips; convert annotations to pinned RTTM/UEM |
| LibriCSS | Repository MIT; source LibriSpeech audio CC BY 4.0 | English, eight speakers, controlled 0–40% overlap | Confirm generated-mixture redistribution and derive RTTM/UEM from source schedules |

The first qualifying set should include at least two languages, 1/2/4/8-speaker cases, silence-only and no-overlap controls, 10–40% overlap, clean and far-field speech, exact-end and padded-tail windows, and clips near the current 137-second cap. Long-file qualification needs a new bounded multi-segment clustering design; the present Community durable stage rejects longer inputs before inference.

## Gates

For every admitted clip:

1. Canonical audio, RTTM, UEM, Go result and source-reference result hashes match the manifest.
2. Safe `RejectAmbiguousTies` either succeeds or returns the recorded ambiguous frame; explicit `LowestIndexTies` is a separate policy identity.
3. Go/source-reference DER and JER deltas pass the reviewed per-case thresholds at 0 s and 0.25 s collars, with overlap included.
4. Repeated same-mode results are byte-identical.
5. No clip is omitted, truncated or silently replaced after a failure.

Absolute DER/JER acceptance thresholds will be set after representative corpus slices are admitted and scored. The existing single-sample values must not define a general quality threshold.
