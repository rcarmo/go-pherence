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

## Public-corpus inventory

No candidate below has been downloaded or run in this checkpoint. The official pages were rechecked on 12 September 2026; "usable" means suitable for a future local, hash-pinned qualification slice, not permission to redistribute a derived subset.

| Corpus | Official terms/source | Verified coverage | Inventory decision |
|---|---|---|---|
| AMI Meeting Corpus | [University of Edinburgh download/licence](https://groups.inf.ed.ac.uk/ami/download/) publishes corpus and annotations under CC BY 4.0 | English meeting speech, multiple microphone conditions and overlap annotations | Primary English far-field/no-overlap candidate. Select bounded scenario meetings only after pinning the exact downloadable files, channels, RTTM conversion and attribution text. |
| VoxConverse | [Official repository](https://github.com/joonson/voxconverse) states CC BY 4.0 but says copyright remains with original video owners | English broadcast/interview audio, varied speaker counts and overlap | Annotation/reference candidate only until each selected video's acquisition and redistribution rights are reviewed. Do not copy video-derived audio into repository artifacts by relying on the annotation licence alone. |
| AliMeeting SLR119 | [OpenSLR 119](https://www.openslr.org/119/) identifies CC BY-SA 4.0 and publishes Train/Eval/Test archives | Mandarin, 2–4 participants, 15–30 minute meetings, near-field and 8-channel far-field audio, varying overlap | Primary Mandarin 2/4-speaker and far-field candidate. Use Eval/Test excerpts only after recording archive/member hashes and the exact speaker-activity-to-RTTM conversion. Share-alike obligations apply to redistributed derivatives. |
| AISHELL-4 SLR111 | [OpenSLR 111](https://openslr.org/111/) identifies CC BY-SA 4.0 and publishes a 5.2 GB test archive | Mandarin real meetings, 4–8 speakers, 8-channel arrays, overlap/noise/quick turns | Primary 4/8-speaker stress candidate. Download is intentionally deferred; later select test excerpts and pin channel choice, archive/member hashes, RTTM/UEM conversion and attribution. |
| LibriCSS | [Official LibriCSS repository](https://github.com/chenzhuo1011/libri_css) provides generation/evaluation tooling; source [LibriSpeech SLR12](https://www.openslr.org/12/) is CC BY 4.0 | English, eight speakers, controlled 0/10/20/30/40% overlap | Strong controlled-overlap candidate, but generated-mixture/audio redistribution terms must be resolved from both LibriCSS tooling and LibriSpeech sources before marking a case qualifying. |

### Required first slice manifest

The first admitted manifest expansion must include these non-substitutable roles, each represented by an exact corpus/session/channel/time interval rather than a corpus-level promise:

| Role | Required evidence | Preferred source |
|---|---|---|
| `en-clean-no-overlap` | 1–2 speakers, clean/close-talk, 0% overlap, one exact-end and one padded-tail window | AMI close-talk or LibriCSS 0S |
| `en-overlap-10-20` | 2+ speakers and independently computed overlap ratio in `[0.10,0.20]` | LibriCSS or rights-cleared VoxConverse |
| `en-overlap-30-40` | eight speakers and overlap ratio in `[0.30,0.40]` | LibriCSS |
| `zh-two-speaker` | Mandarin, exactly two annotated speakers, near/far pair where available | AliMeeting Eval/Test |
| `zh-four-speaker-far` | Mandarin, exactly four speakers, array/far-field channel | AliMeeting Eval/Test |
| `zh-eight-speaker-far` | Mandarin, eight speakers, array channel and natural overlap | AISHELL-4 test |
| `silence-control` | generated digital silence with public-domain/CC0 construction recipe, 0 speakers | local deterministic fixture |
| `cap-boundary` | one clip at the greatest admitted duration and one rejected sample beyond it | any qualifying corpus plus deterministic padding |

Every selected case must record corpus version/retrieval date, immutable upstream URL, archive SHA-256, member path/SHA-256, source sample rate/channels, selected channel/downmix rule, canonical WAV SHA-256, interval start/end in source timebase, language, annotated speaker count, overlap seconds/ratio, RTTM and UEM conversion command/version, licence URL/SPDX/attribution/share-alike status, and whether audio/result redistribution is allowed. The generated Go/reference result hashes remain separate fields.

`PlanDiarizationWindows` permits at most 128 windows, so the largest admitted extent is `window + 127*step` samples (137 seconds for the current 10-second/1-second profile). One additional sample adds a 129th padded window and must fail before inference. The first qualifying set should cover both exact-end and padded-tail behavior near that bound. Longer-file qualification needs a bounded multi-segment clustering design.

## Gates

For every admitted clip:

1. Canonical audio, RTTM, UEM, Go result and source-reference result hashes match the manifest.
2. Safe `RejectAmbiguousTies` either succeeds or returns the recorded ambiguous frame; explicit `LowestIndexTies` is a separate policy identity.
3. Go/source-reference DER and JER deltas pass the reviewed per-case thresholds at 0 s and 0.25 s collars, with overlap included.
4. Repeated same-mode results are byte-identical.
5. No clip is omitted, truncated or silently replaced after a failure.

Absolute DER/JER acceptance thresholds will be set after representative corpus slices are admitted and scored. The existing single-sample values must not define a general quality threshold.
