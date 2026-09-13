# AMI qualifying-corpus pilot contract — 13 September 2026

This checkpoint adds a deterministic source-only path for preparing a real multi-speaker qualification fixture from the official AMI Meeting Corpus. It does not commit corpus media or claim model quality.

## Source contract

- Corpus: AMI Meeting Corpus, official University of Edinburgh distribution.
- Licence: CC BY 4.0, <https://groups.inf.ed.ac.uk/ami/corpus/license.shtml>.
- Evaluation meeting: `ES2004a` (AMI scenario evaluation partition `SC`).
- Audio source: `ES2004a.Mix-Headset.wav`, official headset mix, SHA-256 `3e2560b19bee6952c7c7ce041b0f1ea8a7ea9468044c4eea79d2a2c67e24ab0f`.
- Annotation source: `ami_public_manual_1.6.2.zip`, SHA-256 `b56e5babb2496b8795deeeda7e71178d7fbc9963f94276cf2a3f4b56ebbc9f9d`.
- Official source URLs are recorded in `source.json`.

The preparer reads only the eight allowlisted `segments/` and `words/` XML members for speakers A–D, validates NXT root identities, rejects unsafe archive members, requires exact source hashes, and refuses output overwrite.

## Pilot excerpt

The fixed interval is `350–410 s`. Annotation analysis selected it because all four speakers contribute more than five seconds and it contains substantial overlap. A fresh run produced:

- canonical PCM WAV: mono signed 16-bit, 16 kHz, 960,000 samples;
- 28 clipped RTTM turns across speakers A–D;
- 179 timed lexical words with source IDs and speaker labels;
- exact hashes recorded in `pilot-result.json`.

Generated audio, RTTM and word JSON stay in ignored local storage. The committed result is a reproducibility contract, not redistributed corpus content.

## Verification

```sh
python3 -m unittest scripts/test_prepare_ami_corpus.py
python3 scripts/prepare_ami_corpus.py \
  --meeting ES2004a \
  --audio /path/to/ES2004a.Mix-Headset.wav \
  --audio-sha256 3e2560b19bee6952c7c7ce041b0f1ea8a7ea9468044c4eea79d2a2c67e24ab0f \
  --annotations /path/to/ami_public_manual_1.6.2.zip \
  --annotations-sha256 b56e5babb2496b8795deeeda7e71178d7fbc9963f94276cf2a3f4b56ebbc9f9d \
  --start 350 --duration 60 --output /new/output/directory
```

Next qualification work must run the pinned Community and Whisper pipelines on this fixture, retain model/result hashes, and score corpus DER/JER and speaker-attributed words. One 60-second meeting excerpt is still a pilot; it cannot establish broad-corpus qualification by itself.
