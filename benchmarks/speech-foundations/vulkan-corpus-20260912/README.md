# Public-language, silence and multi-window diagnostics

CPU and Vulkan Tiny inference agree on all tested outputs, but the Portuguese/French fixtures have poor recognition quality. Digital silence also hallucinates text under the unchanged default policy. The new opt-in `SkipDigitalSilence` prevents inference only for entirely zero-valued PCM windows; it does not establish general no-speech detection.

## Public fixture provenance

MINDS-14 by PolyAI, dataset revision `40ce77cb32a384e4d50a568e1ec39ac804019d33`, CC-BY-4.0. Cite: Gerz et al., “Multilingual and Cross-Lingual Intent Detection from Spoken Data”, 2021, https://arxiv.org/abs/2104.08524. Dataset card: `https://huggingface.co/datasets/PolyAI/minds14/blob/40ce77cb32a384e4d50a568e1ec39ac804019d33/README.md`; fetched card SHA256 `3dd23ca4c7834f66ca043847996dbd7c9ca34c846749d86431a9fa19898f4f88`.

The audio and dataset reference text came from retained public dataset-viewer artifacts in `projects/whisper-stt/benchmarks`. No fresh live audio download was required; local source hashes and exact sample extents were verified. Original recordings are mono 8-kHz WAV and were decoded/resampled through the temporary FFmpeg adapter to mono 16-kHz s16 PCM. These clips are not TTS or private recordings.

| Fixture / dataset row | Source SHA256 | Canonical samples / duration |
|---|---|---|
| pt-PT/train/0 | `fc084982ad50c6ea6cf066f08374b9b3aaa628d9a9accb167be5ae9376dbd275` | 144726 / 9.045375 s |
| pt-PT/train/1 | `aacee91914f902b0949425ee29ac984c48a34df55e5c195e65e0c8ff66977484` | 75094 / 4.693375 s |
| fr-FR/train/0 | `84defdc828ef59cec10364354fbc284bc2cc683fdd4a5edd5863b7bb2c6123a8` | 60074 / 3.754625 s |

Asset URL pattern: `https://datasets-server.huggingface.co/assets/PolyAI/minds14/--/40ce77cb32a384e4d50a568e1ec39ac804019d33/--/<locale>/train/<row>/audio/audio.wav`. Source transcripts and full URLs are in [fixtures.json](fixtures.json). Test output and punctuation normalisation are retained separately; references were not edited to match Tiny.

## Quality results and holds

All six fixtures use the same pinned multilingual Tiny weights/config/tokenizer/generation as earlier checkpoints, explicit language (`pt`, `fr` or `en`), timestamp decoding and a 96-token cap. No temperature fallback, no-speech probability rule or language auto-detection is implemented in this path.

| Fixture | Word edits / reference words | WER | Result |
|---|---:|---:|---|
| Portuguese row 0 | 16 / 20 | 80% | Quality hold |
| Portuguese row 1 | 7 / 9 | 77.78% | Quality hold |
| French row 0 | 2 / 5 | 40% | Quality hold |
| JFK with 5 s leading/5 s trailing silence | 0 / 22 | 0% | Padded speech fixture passes |
| 63 s, JFK at 2 s and 42 s; default | 1 / 44 | 2.27% | Hallucinated “you” in final silent window |
| Same 63 s, exact-zero skip enabled | 0 / 44 | 0% | Speech retained; final window empty |
| 5 s digital silence; default | 1 / 0 | Undefined | Hallucinated “you” |
| 5 s digital silence; exact-zero skip | 0 / 0 | Undefined | Empty callback, no segments |

Examples of poor recognition: Portuguese row 1 becomes “Com faspore transfridneir pra minha conta.”; French becomes “Josuait changer mon adresse.” These errors are identical on CPU and Vulkan. That localises the difference away from backend selection for these fixtures, but does not prove a Tiny-model limitation rather than a shared frontend/decoder/policy issue. An external model oracle and/or a larger checkpoint comparison is still required before attributing the cause.

The original JFK source/hash is pinned in [the earlier public-speech report](../vulkan-public-speech-20260912/README.md). Generated compositions preserve its PCM samples exactly, adding only digital zeros. They are synthetic compositions, not natural long-form or multi-speaker audio. The 63-second case uses three non-overlapping 30-second input windows; the final window has only three seconds of real zero-valued PCM and 27 seconds of padding. No word-level overlap reconciliation or multi-speaker coverage is tested.

## Exact-zero option

`PCMTranscribeOptions.SkipDigitalSilence` defaults to false. When enabled, the checked PCM path scans the window before feature extraction. If every sample compares equal to zero, including signed zero and padding, the normal empty window callback is emitted without frontend/encoder/decoder work. Model/tokenizer/config/geometry validation still runs before source access. Nonzero subnormals, tiny finite values, NaN and infinity are not treated as silence; normal validation/inference receives them. The scan checks cancellation every 16384 samples and at entry/exit.

This option neither trims timing nor suppresses mixed speech/silence windows. It has no RMS/energy threshold and makes no quiet-speech/VAD claim. A noisy silent recording may still hallucinate. Existing defaults and other APIs are unchanged.

Unit tests use a deliberately poisoned encoder to verify inference is skipped for exact zeros and attempted otherwise. They cover signed zero, smallest subnormal, NaN/Inf, cancellation, callback/window continuity and no bypass of required decoder validation.

## Native and offline verification

Three final native batches pass three test events with zero failures/skips: 48 inference calls and 72 emitted windows across six fixtures plus the two optional-skip comparisons. Both backends complete every expected window, with exact text/token/timestamp/window equality. Matching failures cannot pass the harness; an error or missing window is fatal. Quality metrics are diagnostics, so the test's pass status means parity/completion and narrow silence-option assertions, not Portuguese/French quality approval.

Offline focused tests: ten top-level tests, 14 passing events; 30 shuffled repeats: 420 events. Broader speech/affine/media/Vulkan regressions, affected vet and amd64 builds pass. Full-tree/backend and Whisper arm64 FFT errors match prior baselines; race build still lacks `gcc`.

Source review found no blocking silence-scan or validation-bypass issue. Fixture metadata checks were expanded after review to validate names, language, basename, SHA encoding and uniqueness. The parent also strengthened inference completion checks. Native output is logged in separate bounded summary/window records. The earlier diagnostic used a long JSON record that Go split into chunks; extracting it requires joining `Output` fragments.

The first native attempt stopped because the hand-entered French canonical sample count was 60078; inspecting the original 30037 samples at 8 kHz established the correct 60074. This was corrected before final runs. The failed attempt and default hallucination diagnostics remain in the report. An attempted FLEURS viewer lookup timed out; no FLEURS audio was used.

## Resources and reproduction

FFmpeg 8.1.2 and the forced Intel ICD were held constant; GOMAXPROCS/CPU worker budget remained 2. These were ordered diagnostic runs, not balanced speed benchmarks. Per-call durations are retained but no comparative speed claim is made. LLM and both speech services stayed inactive/MainPID zero. Before/after swap counters stayed `pswpin=36292`, `pswpout=635627`; snapshots do not establish continuous pressure/thermal behaviour.

```sh
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/intel_icd.x86_64.json
export VK_ICD_FILENAMES="$VK_DRIVER_FILES"
export GO_PHERENCE_VULKAN_DEVICE=Iris
export GO_PHERENCE_WHISPER_TINY_DIR=/path/to/verified/whisper-tiny-169d4a4
export GO_PHERENCE_WHISPER_JFK_PATH=/path/to/pinned/samples/jfk.wav
export GO_PHERENCE_MINDS_FIXTURE_DIR=/path/to/verified/MINDS-cache
GOMAXPROCS=2 CGO_ENABLED=0 make speech-vulkan-corpus-check
```

[Evidence](evidence.json), [metrics](metrics.json), logs and resource snapshots are included. No audio/model weights are committed. Multilingual recognition quality, general no-speech policy, natural long files, overlap/diarization, turbo/quantisation, device-loss recovery and whole-job gates remain unfinished. Strict SincNet retains four failures. Nothing was pushed, deployed or restarted.
