# Trained Turbo F32 quality and performance baseline

The Go/Vulkan large-v3-turbo path completed the four pinned public clips with zero word edits in three final runs. Its current F32 encoder is slow: each request takes about 15–16 seconds, dominated by roughly 14.25 seconds of encoder execution. Quality improved on these fixtures; the performance target is not met.

## Model and resource admission

Checkpoint: `openai/whisper-large-v3-turbo`, revision `41f01f3fe87f28c78e2fbf8b568835947dd65ed9`, `model.safetensors` SHA256 `542566a422ae4f3fd23f1ba11add198fca01bbf82e66e6a2857b3f608b1eb9d1`, 1,617,824,864 bytes. The downloaded file matched the repository linked-object hash. All config/tokenizer/generation hashes match the existing generation manifest; the test rechecks all four assets before loading.

The safetensors file has 587 F16 tensors, 68,696 header bytes and `format=pt` metadata. `LoadConfiguredModelSourceChecked` validates metadata and widens values into owned F32 tensors, then the mmap source is closed. Total model payload expanded to F32 is 3,235,512,320 bytes. This checkpoint does not implement native F16 inference or quantised kernels.

Model-card source: `https://huggingface.co/openai/whisper-large-v3-turbo/blob/41f01f3fe87f28c78e2fbf8b568835947dd65ed9/README.md`, SHA256 `aaef74a740faca90fa1899c4233ebe17f2093b9846d0acf16d9131bf650e9585`; the card declares MIT. Weights and card cache remain outside the Go checkout, under `projects/models/whisper-turbo-41f01f3`; no model weights are committed or attached.

The native test installs a temporary package memory budget of 4 GiB and 40 allocations, then restores the prior budget. This caps explicit `VkDeviceMemory`, not all process/driver memory. The resident full encoder uses 34 allocations totalling 2,641,735,680 bytes (logical weights 2,548,039,680 plus scratch 93,696,000). Model width 1280, FFN width 5120, 20 heads, 32 layers, 3000 frames, 34 plans and 390 stages. Host encoder weights are released after upload; the Go CPU decoder remains resident.

The LLM and both speech services stayed stopped. One-second monitoring over 261 samples across the three final runs recorded minimum MemAvailable of 21,516,684 KiB (about 20.52 GiB); `pswpin=36297` and `pswpout=635627` stayed unchanged. The swap-in counter is five pages above earlier historical checkpoints before this monitor started; this report does not claim the whole day had no swap activity. Resource snapshots and sampled memory are not full allocation/energy/thermal telemetry.

## Numerical and public speech results

A short four-frame input exercises the complete trained width/depth against direct scalar reference boundaries. All 34 boundaries per run pass the predeclared `2e-3 + 5e-4*abs(reference)` budget. Across three final runs: 102 comparisons, 261,120 output values, maximum absolute error `2.5033950805664062e-5`. This small sequence does not establish full-length tensor parity.

Public speech uses the same FFmpeg adapter/canonical PCM and references as the Tiny tests, explicit language, timestamps, pinned generation policy and 96-token cap. There is no CPU-encoder fallback. Each request passes exact PCM extent, finite output and completed nonempty single-window checks. WER is reported independently rather than equating test completion with corpus acceptance.

| Fixture | Turbo edits / words | Tiny edits / words | Turbo inference range |
|---|---:|---:|---:|
| Portuguese row 0 | 0 / 20 | 16 / 20 | 15.756–15.840 s |
| Portuguese row 1 | 0 / 9 | 7 / 9 | 15.197–15.265 s |
| French row 0 | 0 / 5 | 2 / 5 | 15.139–15.163 s |
| JFK English | 0 / 22 | 0 / 22 | 15.708–15.787 s |

All 12 final transcripts have zero edits; token IDs, timestamps and window metadata are identical across the three runs. These total 56 reference words per four-clip set, 168 with repeats; not an independent 168-word corpus. Tokens and generated VTT/JSON are retained. This is a four-clip quality improvement, not broad multilingual acceptance, DER evidence or independent external Turbo-oracle parity.

Example Portuguese row 1: “Como faço para transferir dinheiro para a minha conta?” French: “Je souhaite changer mon adresse.” The unchanged Tiny failures remain historical outcomes of that smaller model under the matched policy.

## Performance hold

Recorded inference time surrounds checked PCM reading/frontend, Vulkan encoding, CPU cross-KV/decoder and callback mapping. It excludes model hash/load, FFmpeg preparation and resident construction. These runs were not balanced against another Turbo backend. Construction took about 1.6 seconds and checked F16→F32 loading about four seconds in the initial runs; hashes and token loading are separate.

A separate diagnostic re-executes Portuguese row 0, checks exact output against the PCM API and reports stage times. Across three final runs the encoder alone takes 14.24–14.26 seconds. Frontend, cross-KV and decoder durations plus 30 decoder calls are in `metrics.json`. These are host-wall stage timings, not GPU timestamps or a decomposition of the median public-request run.

Each short clip still requires the fixed 30-second encoder input. Full-length F32 compute dominates. Possible F16/quantised execution, tiling/dispatch optimisation and safe model-length policies require separate correctness and performance evidence. The current request times do not reach even 1× realtime for these four clips, much less the planned ≥8× ASR target. No default or deployed service was changed.

## Test harness and review

`TestVulkanTurbo` is default-disabled and requires an expected GPU name, pinned local model and a Go timeout no longer than five minutes. Public speech is separately enabled; a fixture selector permits one clip without unrelated fixture paths. Native tests stop after failure. Native encoder close checks live allocation return; budget restoration is an independent cleanup callback.

Review found two initial harness issues: a fatal leak check could bypass budget restoration, and a single selected fixture required both source directories. Separate cleanup callbacks and selected-file validation fixed these before final runs. Offline selector tests cover absent/unrelated paths and unknown selections. The first compile also caught the safetensors metadata field spelling `DType`; this was fixed before native execution.

Three final native processes each pass the short-reference and four-clip speech subtests: nine total test/subtest pass events, zero failures/skips. Separate initial short/speech runs and a standalone short-reference make run are retained outside these totals. Offline focused tests pass 11 top-level/15 events, plus 30 shuffled repeats (450 events). Full speech/Vulkan/affine/media regressions, affected vet and amd64 builds pass. Full-tree/backend and Whisper arm64 FFT failures match prior baselines; race compilation still lacks `gcc`.

## Reproduction and evidence

Only run within a coordinated compute window with adequate memory and verified stopped services. The test performs no downloads.

```sh
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/intel_icd.x86_64.json
export VK_ICD_FILENAMES="$VK_DRIVER_FILES"
export GO_PHERENCE_VULKAN_DEVICE=Iris
export GO_PHERENCE_WHISPER_TURBO_DIR=/path/to/verified/whisper-turbo-41f01f3
GOMAXPROCS=2 CGO_ENABLED=0 make speech-vulkan-turbo-check

# Public speech and separate stage timing diagnostic:
export GO_PHERENCE_MINDS_FIXTURE_DIR=/path/to/MINDS-cache
export GO_PHERENCE_WHISPER_JFK_PATH=/path/to/pinned/samples/jfk.wav
GO_PHERENCE_TEST_VULKAN_TURBO_SPEECH=1 GO_PHERENCE_TEST_TURBO_STAGES=1 \
GOMAXPROCS=2 CGO_ENABLED=0 make speech-vulkan-turbo-check
```

[Evidence](evidence.json), [metrics](metrics.json), model/source hashes, raw test logs and memory samples are included. General no-speech policy, natural long files, overlap/diarization, quantised/F16 execution, device-loss recovery and whole-job targets remain unfinished. Strict SincNet retains four failures. Nothing was pushed, deployed or restarted.
