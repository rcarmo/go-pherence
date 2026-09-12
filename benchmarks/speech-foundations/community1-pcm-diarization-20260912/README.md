# Experimental end-to-end Go diarization

The Go PCM pipeline produces the same full and exclusive turns as a fresh pinned pyannote reference on the public30-second tutorial sample, using an explicitly selected lowest-index reconstruction tie policy. Full DER matches at both tested collars. Strict tie handling and previously recorded neural intermediate gates still fail; production qualification is incomplete.

## Connected path

`ExperimentalDiarization` connects the checked experimental segmentation and embedding wrappers to existing mask selection, clustering admission, AHC, raw-PLDA preparation, VBx, centroid assignment, overlap-add reconstruction and full/exclusive turns. The caller supplies immutable models, a canonical mono16k absolute-sample reader, exact input length, CPU modes and policy. `media.PCMReader` implements the reader contract. No media decoder, service, driver or external neural runtime is invoked by Go inference.

The window plan follows pinned `Inference.slide`: complete `unfold(window,step)` windows, then one zero-padded orphan only when input is shorter than the window or `(samples-window)%step != 0`. The orphan starts at `completeWindows*step`. Exact-end inputs do not receive a duplicate window. Empty inputs are rejected. The experimental cap is128 windows and four hours of source extent; at the tested10s/1s geometry this is far below four hours. This is not a production long-file scheduler.

Reads must fill the declared valid extent. EOF accompanying a full read is accepted, including wrapped EOF; short reads and internal truncation fail instead of becoming padding. One zeroed window buffer is reused. Models/results retain no reader or caller-PCM alias. No result escapes errors or cancellation; observers may already have received progress notifications.

Each window is segmented, hard-powerset decoded, and passed to `SelectEmbeddingMasks`. Clean speech is chosen only when strictly above the configured minimum; otherwise all speech is used. Fbank/CNN run once per window, with private owner-bound features reused across all local masks. Empty-mask projection bias is retained as numerical output; training admission separately requires the existing0.2 clean-speech ratio. Overlap-only input can fail with `ErrNoTrainingEmbeddings` rather than inventing a centroid.

Retained diagnostics include window starts/valid samples/padding, segmentation, embeddings, pre-resize selected support, post-resize mask support and postprocessing stages. Turns use the existing canonical frame-centre grid. They are not clipped to input length, mapped to container PTS/edit lists, renamed or centroid-reordered. Positive gap filling retains the existing semantics. Unsupported KMeans and undefined cosine cases remain errors.

## Pinned policy and PLDA

The trained run uses10-second windows/1-second steps, producing21windows ×589segmentation frames ×3local slots. The nominal grid is270-sample step,991-sample receptive-field size and495-sample first centre. Existing reconstruction resets frame start to canonical0, matching source aggregation.

The embedding minimum obtained with the source's exception-based search is400samples. At this length unweighted statistics can be NaN without an exception; this is a source-compatible overlap-clean selection threshold, not a claim that400samples yield a valid unweighted embedding. The pipeline processes10-second windows and separately filters training rows.

Policy: overlap exclusion enabled, automatic1–64speakers, AHC threshold0.6, Fa0.07, Fb0.8, min-duration-off0, constrained assignment enabled. An initial diagnostic mistakenly selected unconstrained assignment; that run is retained and is not the final source-policy comparison.

PLDA uses the pinned checkpoint's mixed-dtype NPZ tensors with explicit dimensions256→128→128:

| Asset | SHA256 |
|---|---|
| `xvec_transform.npz` | `325f1ce8e48f7e55e9c8aa47e05d2766b7c48c4b25b8de8dd751e7a4cc5fbe8f` |
| `plda.npz` | `9b77bcd840692710dd3496f62ecfeed8d8e5f002fd991b785079b244eab7d255` |

Raw preparation passes its fixed conditioning policy: infinity-norm condition56.00115466461734, two-sided inverse residual1.7763568394002505e-15. Sorted raw PLDA coordinates are not asserted elementwise equal to SciPy eigenvectors; downstream behaviour is compared. Model licences and existing implementation attributions remain separate in `NOTICE`; raw checkpoint/PLDA weights are not included here.

## Full public-sample result

The final constrained Go run has37training rows,2clusters,13full turns and12exclusive turns. Its1779×2 activity timeline reports84ambiguous frames under `LowestIndexTies`.

A fresh one-thread, MKLDNN-off, batch-size1 pyannote source reference uses the same converted weights, source-pinned frontend/configuration and public PCM. It produces:

- Exact agreement for all37,107binary segmentation values.
- Maximum absolute embedding error3.3080577850341797e-6 over16,128values, below the explicit2e-4 comparison gate.
- Matching full and exclusive speaker turns after one global label mapping, with boundary tolerance1e-12seconds.
- Repeated reference segmentation/embedding NPY arrays that are byte-identical. Two final-policy Go results are byte-identical, including retained diagnostics and turns.

DER scoring uses the pinned RTTM, explicit UEM `[0,30]`, overlap included, and optimal speaker mapping:

| Output | Collar | Go DER | Fresh reference DER | Delta |
|---|---:|---:|---:|---:|
| Full | 0s | 5.2073921971% | 5.2073921971% | 0pp |
| Full | 0.25s | 1.3177976791% | 1.3177976791% | 0pp |
| Exclusive | 0s | 10.2909394251% | 10.2909394251% | 0pp |
| Exclusive | 0.25s | 4.9445005045% | 4.9445005045% | 0pp |

Exclusive output is scored against the same overlapping reference, which explains its larger missed-speech contribution. The saved historical pyannote WAV/FF result also has identical full DER, but the fresh reference is the current comparison. The scorer's `--require-reference-parity` checks masks, embedding tolerance, turn mapping and full DER, writes evidence first, then exits nonzero on failure.

## Retained failures and limits

- Default `RejectAmbiguousTies` fails at frame397 with the corrected constrained policy. The initial unconstrained run failed at frame450. `LowestIndexTies` is explicitly selected for diagnostics; it is not asserted to match NumPy tie identities on other inputs.
- All prior segmentation and embedding strict intermediate failures remain. This increment changes no Fbank, convolution, recurrent, pooling or clustering arithmetic and widens no tolerance.
- The initial reference script used an incorrect PLDA constructor keyword; this failed before inference and was corrected. The initial Go harness used an incorrect reader accessor; its build failure is retained.
- One public30-second source is not general DER, multilingual speaker coverage, natural long-file, large-speaker or robustness qualification. Padded tails are tested synthetically, not against a broad public end-to-end corpus.
- The elapsed Go test process was roughly48seconds for30seconds of audio. These are diagnostic process durations, without balanced timing/resource/energy measurement; the planned whole-job performance target is unmet.
- No production CLI/HTTP/job workflow, durable scheduling, recovery, media switch or service deployment is added. FFmpeg remains the temporary/default media backend; go-264 remains unmerged.

## Verification

- `make speech-foundations-check speech-sincnet-fma-check speech-media-integration`: pass.
- Community-1/audio/media/SIMD regressions:710passing events /320top-level passes; seven explicit opt-in test skips plus the no-tests SIMD package skip.
- Thirty shuffled window/wrapper repetitions:120passing tests, zero failures/skips. Forced AVX2/FMA-off wrapper checks:4pass.
- Tests cover exact-end/orphan/one-sample plans,128-window cap, invalid-before-read policy, manual neural/mask/postprocess composition, silence/single-row/clustered paths, overlap-only admission failure, explicit speaker-count override, no-clobber source ownership, short/wrapped-EOF reads, all progress-stage cancellation/error injection and recovery.
- Three script contract tests pass. Fresh-reference scorer parity passes; a deliberately modified embedding copy fails while retaining its evidence.
- Affected vet and arm64 cross-build pass. Full-tree build retains exact baseline failures. Race compilation still lacks `gcc`.

A delegated source-only wrapper review found no concrete ownership/cancellation issue. Added an explicit segmentation-result allocation bound and clarified must-fill reader semantics. Tie policy was already validated before reads by reconstruction preflight. Suggestions to restrict explicit NumSpeakers to min/max or guarantee minimum speech were not adopted: they conflict with source count-override and clean-mask fallback semantics; tests/documentation now cover those decisions. An earlier delegated source audit timed out without findings.

## Reproduce

Use a coordinated CPU window. Required converted model directories and raw PLDA/public assets are hash-checked by the test.

```sh
export GO_PHERENCE_COMMUNITY1_SEGMENTATION_DIR=/path/to/segmentation-cache
export GO_PHERENCE_COMMUNITY1_EMBEDDING_DIR=/path/to/embedding-cache
export GO_PHERENCE_COMMUNITY1_PLDA_DIR=/path/to/pinned/plda
export GO_PHERENCE_COMMUNITY1_PUBLIC_WAV=/path/to/pinned/pyannote-sample.wav
export GO_PHERENCE_COMMUNITY1_DIARIZATION_OUTPUT=/path/to/new/go-result.json
GOMAXPROCS=2 CGO_ENABLED=0 make speech-community-diarization-lowest-ties
# Expected tie rejection until qualified tie handling is implemented:
unset GO_PHERENCE_COMMUNITY1_DIARIZATION_OUTPUT
GOMAXPROCS=2 CGO_ENABLED=0 make speech-community-diarization-check
```

`community1_pipeline_reference.py --help` documents the explicit local reference assets. `score_community1_diarization.py --help` documents saved-only scoring. Neither is used at runtime by Go. [Evidence](evidence.json), [DER result](der-final.json), compressed Go results and reference metadata retain the tested scope. Raw reference tensor arrays stay in the model cache. Services remain inactive; no push, deployment or restart. Compute window: `whisper-community-windows-1008`.
