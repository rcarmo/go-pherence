# First public-speech Tiny qualification

The Go-owned checked PCM pipeline transcribed the 11-second public JFK sample with zero word edits against its 22-word reference. CPU and resident-Vulkan encoding produced identical text, token IDs, timestamps and window metadata in all 54 final transcriptions. This is one English smoke fixture, not multilingual/corpus acceptance.

## Source and inference policy

Input is `samples/jfk.wav` from the pinned whisper.cpp checkout `c44b60b8053bbf2a5c1e014f11323fb3f2485177`, SHA256 `59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e`, 352,078 bytes. Source URL: `https://github.com/ggml-org/whisper.cpp/blob/c44b60b8053bbf2a5c1e014f11323fb3f2485177/samples/jfk.wav`. The recording is an excerpt of President Kennedy's public inaugural address. Audio was read from the existing vendor cache; it is not copied into this report.

Weights/config/tokenizer/generation are the hash-pinned multilingual `openai/whisper-tiny` checkpoint recorded in [trained Tiny evidence](../vulkan-tiny-bridge-20260912/README.md). All four file hashes are checked before inference. No foreign neural runtime is used: native Go exact frontend, F32 CPU or Vulkan encoder, Go CPU cross-KV and decoder, checked timestamp policy and window mapping.

The temporary FFmpeg/ffprobe adapter is used for both arms. FFmpeg 8.1.2 produces exactly 176,000 mono 16-kHz s16 PCM samples; this count/rate is asserted, not only logged. Decode selects the probed audio stream, disables MOV external references, restricts protocols to `file,pipe`, strips non-audio streams/metadata and uses `pcm_s16le`, `-ac 1`, `-ar 16000`, two threads and the existing duration/output caps. Both inference arms read the same decoded immutable WAV. The input is already canonical WAV, so this fixture does not qualify lossy formats or resampling accuracy.

Generation: explicit `en`, transcribe, checked timestamps, pinned generation suppression/initial-timestamp policy and MaxNewTokens 96. The 11-second source occupies one 480,000-sample window with 304,000 zeros at the tail. The model receives its full 3,000-frame input. CPU/GPU comparison changes only the optional resident encoder.

CPU settings: GOMAXPROCS 2, linear workers 2, row block 32, INT8/F16 attention disabled, NVIDIA disabled. This is the current CPU backend under that explicit thread budget; it is not an optimally tuned all-core CPU baseline. Vulkan uses Iris Xe RPL-P through the forced Intel ICD. Model/driver/executable hashes and resource snapshots are recorded.

## Transcript and timestamps

> And so my fellow Americans ask not what your country can do for you, ask what you can do for your country.

One segment: **0.00–10.50 s**, 24 text token IDs. Every final transcript and token sequence is identical across arms/repeats. Timestamp checks require monotonic, nonempty segments within the actual 11-second extent; this does not establish alignment error against manually annotated word boundaries.

Reference text was declared before execution: “And so my fellow Americans ask not what your country can do for you ask what you can do for your country”. The scorer lowercases, strips punctuation/apostrophes, separates other boundaries and uses word-level Levenshtein distance. The denominator is asserted to be 22 words. The predeclared fixture gate is WER ≤15%; measured WER is **0/22 = 0%** in every run. No general WER budget was ratified by this smoke test.

The normaliser is unit-tested for insertion, deletion, substitution, case, punctuation, apostrophes, empty strings and Unicode. It is a fixture utility, not a multilingual evaluation standard.

## Warm equal-output comparison

Each of three final batches makes one initial validated call per arm, then four alternating ABBA/BAAB blocks (8 warm samples per arm). Totals: 54 transcriptions, 48 warm timed samples, six initial calls. Every call validates WER, timestamps and exact output equality. Initial and exploratory runs are recorded separately.

| Batch | CPU median | Vulkan-encoder median | CPU / Vulkan | Audio / Vulkan time |
|---|---:|---:|---:|---:|
| 1 | 1172.3 ms | 583.7 ms | 2.008× | 18.84× |
| 2 | 1164.8 ms | 586.2 ms | 1.987× | 18.77× |
| 3 | 1167.9 ms | 608.7 ms | 1.919× | 18.07× |

Times surround `TranscribePCMWindows`: bounded PCM reads, exact frontend, encoding, cross-KV, autoregressive decoding, timestamp/window mapping and callback storage. They exclude FFmpeg preparation, hash verification, model/tokenizer loading and Vulkan construction. Weights and the canonical file are warm; decoder KV is constructed on each call. The audio/time multiple is only this warm 11-second fixture, not the plan's complete-job ≥8× ASR acceptance gate or a cold latency guarantee.

A separate sequential diagnostic matches the exact PCM API result and reports:

| Stage | Three-run range |
|---|---:|
| Exact frontend | 90.9–95.1 ms |
| Vulkan encoder, including upload/download | 300.6–305.0 ms |
| CPU cross-KV | 31.8–39.3 ms |
| CPU decoder/timestamp policy | 184.4–189.1 ms |

These diagnostic calls are outside the alternating comparison and must not be summed as its measured median. They have 29 decoder calls each and 24 emitted text tokens. No GPU timestamp, allocation/GC, energy or continuous thermal attribution was collected.

## Verification and review

Final native batches: three test passes, zero failures/skips. Offline focused selection: 18 top-level tests/77 events; 30 shuffled timestamp/PCM/scorer repeats: 600 events. Existing speech/frontend/loader/generation/media/affine/Vulkan regressions, affected vet and amd64 builds pass. Full-tree/backend and Whisper arm64 FFT failures match prior baselines. Race build still lacks `gcc`.

Review found that the initial harness logged but did not assert PCM extent, did not pin the 22-word denominator and permitted a looser outer timeout. All three checks were tightened before final runs. WER DP, pinned inputs, exact arm comparison and native cleanup were otherwise accepted in the scoped source review. A preliminary compile caught an unused import and the package's two-argument `min` helper; both were corrected before the first native run.

Native testing is explicit opt-in. It requires `go test -timeout` no greater than 120 seconds and an inner 110-second context. It verifies input/model hashes and expected GPU name, fails on errors without fallback, and checks native allocation return after cleanup. No services start in the harness. The LLM and both speech services stayed inactive/MainPID zero; before/after swap counters stayed `pswpin=36292`, `pswpout=635627`. These are snapshots, not proof of zero transient pressure.

## Reproduce

Use an authorised compute window and the existing verified model/audio cache. No downloads or credentials are used by the test.

```sh
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/intel_icd.x86_64.json
export VK_ICD_FILENAMES="$VK_DRIVER_FILES"
export GO_PHERENCE_VULKAN_DEVICE=Iris
export GO_PHERENCE_WHISPER_TINY_DIR=/path/to/verified/whisper-tiny-169d4a4
export GO_PHERENCE_WHISPER_JFK_PATH=/path/to/pinned/samples/jfk.wav
GOMAXPROCS=2 CGO_ENABLED=0 make speech-vulkan-public-speech-check

# Equal-output warm CPU/GPU comparison:
GO_PHERENCE_TEST_SPEECH_CPU_COMPARE=1 GO_PHERENCE_TEST_SPEECH_TIMING=1 \
GOMAXPROCS=2 CGO_ENABLED=0 make speech-vulkan-public-speech-check
```

[Evidence](evidence.json), [transcript](jfk-transcript.vtt), metrics, logs and resource snapshots are included. Multilingual corpora, silence/hallucination policy, overlap, long files, diarization, full-job quality/performance, turbo/quantisation and device-loss recovery remain unfinished. Strict SincNet retains four failures. No runtime/default behaviour was changed in this checkpoint; nothing was pushed, deployed or restarted.
