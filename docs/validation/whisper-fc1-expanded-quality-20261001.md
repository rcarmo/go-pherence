# FC1-only expanded quality and retained profile — 1 October 2026

Expanded recorded-speech testing does **not** qualify the FC1-only integer-dot candidate for general output/timing parity. It matches the retained Go baseline on two additional Portuguese recordings, but changes one word and segment boundaries on a20s English podcast excerpt. With genuine VAD/word alignment, text/segments match the baseline but one shared word boundary moves20ms in every repeat. The mode stays experimental and defaults remain unchanged.

## Scope and independent references

Starting revision: `38db449b966c3d72d849feeabcff9d5c8bb2b691`. Original reference: whisper.cpp `c44b60b8053bbf2a5c1e014f11323fb3f2485177`, original turbo Q5 model SHA256 `394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2`. Candidate quantises FC1 only; FC2, Q/K/V/O, attention and decoder remain the retained F32 paths.

Additional Portuguese audio is MINDS-14 `pt-PT/train` rows1/2, retained by the previous project with dataset URLs pinned to revision `40ce77cb32a384e4d50a568e1ec39ac804019d33`. Human transcriptions come from the retained `minds-pt.json` dataset metadata, not original-engine output. They are supplied labels, not independently audited verbatim transcripts. Audio hashes:

- Row1,4.693375s: `f6624137df0a2a33f1e523eb7ffce50e7d0b5624a6b736772b6f6f0645bcf045`.
- Row2,43.76s: `bda07e92625321318397e41cfa8169839d361c67442befea86a1ef2e047e919f`.
- Podcast first20s: `a68c5cffadd6098504a83b223b003bfb8330fca4198533ee4b861f70a7e23646`. Source WAV pin is retained in provenance. The source is already mono16kHz PCM16; the excerpt copies exactly the first320,000 samples and writes a canonical44-byte WAV header. No resampling/compression/inference generated this fixture.

No human transcript was found for the podcast excerpt. Original-engine word disagreement there is not acoustic WER, and neither version is declared more accurate. No long-form/resume or diarization test occurred.

## Expanded request results

Fresh baseline and candidate, five requests each, same checkpoint/audio/language, four CPU threads, greedy decode, physical Intel Vulkan and F32 key32 flash attention. First requests are retained; no samples discarded. Processes are sequential baseline then candidate, not randomised. Original engine runs once per fixture with matching language/greedy/flash flags, writes full token/segment JSON, and is a diagnostic output oracle—not a five-repeat speed benchmark.

| Fixture | Go baseline median (s) | FC1-only median (s) | Change | Candidate vs Go baseline |
|---|---:|---:|---:|---|
| MINDS PT row1 | 6.444 | 6.111 | −5.17% | All returned windows/segments/content tokens exact |
| MINDS PT row2,43.76s | 14.425 | 13.754 | −4.65% | All returned windows/segments/content tokens exact |
| Podcast20s, no VAD | 9.207 | 8.809 | −4.33% | One word and segment/timestamp decisions differ |
| Podcast20s, native Silero VAD + words | 12.215 | 11.909 | −2.50% | Text/segments exact; shared word boundary differs20ms |

Candidate decisions are identical across its five repeats. No independent overall speed acceptance follows from these warm request comparisons. Original process/phase timings include different precision/loading/context boundaries; their single samples stay separate in raw evidence.

### Human-reference word scoring

Normalisation: Unicode NFKC, `pt-PT` lowercase, punctuation/non-letter/non-digit runs converted to spaces, whitespace word tokens. Standard unit-cost word edit distance; deterministic alignment priority match/substitution/deletion/insertion. Hand-count scorer tests and an independently written rolling-row distance implementation confirm totals. No grammar/spelling repair, filler removal or number rewriting is used.

| MINDS human-labelled fixture | Reference words | Go baseline | FC1-only | Original engine |
|---|---:|---:|---:|---:|
| PT row1 | 9 | 0 edits,0% | 0 edits,0% | 0 edits,0% |
| PT row2 | 47 | 8 edits,17.02% | 8 edits,17.02% | 8 edits,17.02% |

Row2 contains7 substitutions and1 insertion under this normalisation. Exact edits and reference text are retained. This is a two-record supplied-label result, not a representative WER benchmark; ambiguous/filler wording in the label was not acoustically adjudicated. No candidate-specific text degradation appears on these two labels, but neither has perfect reference agreement.

### Original agreement and timing boundaries

Portuguese normalised text matches the original for both additional rows, but timings already differ in the unchanged Go baseline:

- PT row1: original endpoint4.58s; both Go modes return4.693375s.
- PT row2: original first boundary4.96s, both Go modes4.98s. Original final filler ends35.12s; both Go modes return the clipped recording endpoint43.76s. This pre-existing gap fails any claim of general original timing parity.
- Podcast no-VAD: baseline/original contain “I'm not an illogist”; FC1-only emits “I'm not an illiterate”. That is1 substitution across81 normalised original words (1.23% engine disagreement, **not WER**). Baseline has6 segments, original5, candidate6 with different later boundaries. Human correctness is unknown.
- Podcast VAD+words: all segment text/tokens/times remain baseline-exact. The boundary between “and” and “my” shifts11.314→11.294s (word46 end/word47 start) consistently in all five repeats. This is a20ms internal alignment change, not a passed exactness gate. No timing tolerance was widened.

The broader evidence supersedes any inference that the earlier five short fixtures established general FC1 quality. They established only their recorded fixture equality.

## Retained candidate profile

The stage profiler now explicitly admits the FC1-only backend, negotiates integer-dot before loading the device, labels the extra activation-quantisation stages and rejects other unadmitted modes. It uses the complete verified original source, three individually fenced stage passes and requires the same final hidden output as its whole-plan execution. No activation capture is enabled in this profile.

| Operator group, per encoder pass | Mean fenced stage time (s) |
|---|---:|
| F32 key32 attention | 2.162 |
| Packed Q5/F32 FC2 | 1.110 |
| Integer-dot FC1 | 0.725 |
| FC1 activation quantisation | 0.026 |
| Q | 0.299 |
| K | 0.298 |
| V | 0.296 |
| O | 0.291 |
| Convolutions | 0.107 |
| GELU | 0.077 |
| Add | 0.077 |
| Normalisation | 0.066 |

Attention, FC2 and Q/K/V/O remain substantial costs. Individually fenced host timings alter synchronisation and do not replace request/end-to-end measurements. The updated profiler changes only diagnostic admission/labels, not arithmetic/defaults/production selection. No broader quantisation candidate was implemented in this window.

## Verification, isolation and evidence

All native processes retained CPU4/8GiB/no-swap, heap target4GiB, no network/read-only root and existing Intel render node. Qwen idle and host available memory≥6GiB guards stayed active. Qwen LAN/Gemma/services/resources were unchanged. Native test deadlines were≤120s. An outer tool timeout left a completed Portuguese container behind; actual exit0/noOOM was inspected and the container removed before the next run.

A first podcast attempt was rejected by the existing16MiB/60s admission bound; no inference ran. The fixture was reduced to its first20s. FFmpeg was unavailable inside the isolated runner, so the already-canonical PCM was copied directly. Both failures are retained; the bounded gates were not relaxed. A delegated reference search timed out with no independent review approval.

`make model-layout-check host-build host-vet host-test`, affected race tests and ARM64/RISC-V builds pass with visible phase markers and isolated exit0/noOOM. No shader/arithmetic changed, so this window does not claim a new shader qualification. Affected tests/vet and documentation checks pass. Native/build work is drained before resource release.

[Hashed evidence](../../benchmarks/speech-foundations/whisper-fc1-expanded-quality-20261001/) retains compressed supplied audio/excerpt, dataset text/URLs and source pin, raw original/Go JSON, failure/native logs and states, independent scorer check, exact differences, profile samples/totals and runner. The [earlier FC1 report](whisper-integer-dot-fc1-20261001.md) remains the limited-fixture result, not general approval.

## Remaining goal

General FC1 acoustic/word-timing acceptance is **not established**. Keep it experimental and never silently substitute it for F32. A future arithmetic candidate must qualify on this expanded set as well as earlier fixtures. Prefer investigating measured attention/FC2/F32 projection costs under their explicit numerical contracts rather than assuming more quantisation is safe. Matched original Vulkan+flash+VAD workflow speed, independent wider speech/word timing, long-form/resume and fault coverage remain open; the goal is unmet.
