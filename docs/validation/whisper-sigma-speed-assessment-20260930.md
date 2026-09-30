# Can Whisper match Nemotron's speed on Sigma?

**Yes, with an existing alternative backend: Whisper has already matched or exceeded the current Nemotron ASR rate on this hardware. Matching that rate in the current native Go Whisper implementation is plausible but unproven.** No new inference, build or GPU test was run for this assessment; the Decoder pipeline continues separately.

## Existing local evidence

Sigma is an **Intel i5-1340P**, four P-cores with SMT plus eight E-cores, 32 GB shared memory and Intel Iris Xe RPL-P. It supports AVX2, FMA and AVX-VNNI, but not AVX-512, AMX or XMX. The retained model configuration is Whisper **large-v3-turbo**, with 32 encoder layers, width 1280 and four decoder layers.

| Backend and workload | Recorded time | Real-time factor | Limits of comparison |
|---|---:|---:|---|
| Current Nemotron CPU ASR, Decoder 76m47s | 26m12s | **0.341** | Current full-job result; no diarization in that baseline |
| whisper.cpp turbo Q5, Vulkan + flash attention, 43.76s human Portuguese, no VAD | Mean **6.682s** across four balanced runs | **0.153** | Same machine, four threads; short fixture and different audio; includes CLI load |
| Earlier whisper.cpp Vulkan Q5 + VAD application with Community-1, 37m32.5s recording | **10m14.25s** | **0.273** | Full pipeline on a different recording; quality and six speaker labels were not independently assessed |
| faster-whisper CPU INT8 turbo, 43.76s Portuguese | **13.18 / 13.36s**, eight threads | **0.301 / 0.305** | Resident model, excludes load; VAD off, word timestamps enabled; twice the present CPU quota |
| Retained native Go Whisper JFK service job, 11s audio | About **28s** to transcript publication | About **2.55** | Historical job/configuration; not a controlled comparison with the current Nemotron run |

Raw retained evidence: `projects/whisper-stt/benchmarks/gate.json`, `faster-whisper.json`, and the completed job metadata in `projects/whisper-stt/data/jobs.sqlite`. The full-length job is `f5ae9bb4-ea13-4cd6-98bd-26a57fe750ca`, duration 2252.501333s, elapsed 614.24574392s. Only timing/status metadata was read from that private job; its transcript was not copied into this report.

The selected historical Whisper backend used four threads, affinity CPUs 0–7, GPU and flash attention enabled, greedy decoding, Silero VAD and a patch restoring original-timeline token timestamps after VAD. Source `whisper.cpp` revision: `c44b60b8053bbf2a5c1e014f11323fb3f2485177`. The Q5 model and runtime binary still exist locally. The old Whisper services remain stopped as requested.

These results establish that the hardware can run Whisper at the target rate. They do not predict runtime or accuracy on the Decoder recording. Continuous English interview speech may give less VAD benefit than the earlier recordings.

## Why the native Go path was slower

Source review of `projects/go-pherence-transcribe-web/` identifies these costs:

1. **Fixed padded windows.** `model/whisper/pcm_transcribe.go` runs the model over padded windows and uses a roughly 30-second window plan. Recovery subdivisions are padded again. Short clips and tails still incur substantial encoder work. Trimming without preserving the trained padding/attention contract can change outputs.
2. **CPU encoder cost.** Turbo has a 32-layer, 1280-wide encoder. `model/whisper/encoder.go` materialises per-head Q/K/V buffers and attention-score matrices and performs dense attention over the full window. Current packed/FP16/int8 branches already exist; their dispatch and scaling need profiling rather than blanket replacement.
3. **Autoregressive decoding.** `model/whisper/decoder.go` executes token-by-token projections and attention. Self-attention KV is preallocated, and cross-attention K/V is already computed once per window. Reimplementing those caches would duplicate completed work. Reusable scratch and output-projection costs remain candidates.
4. **Word alignment adds work.** The old deployed Go profiles had `word_timestamps=true`. `decodePCMWindow` creates a second decoder state, including cross-KV preparation, and `word_alignment.go` teacher-forces generated text through the decoder. Current Nemotron returns coarse emission chunks without that alignment. Equal-speed comparisons must either disable alignment in both or account for its useful output separately.
5. **Encoder-only offload.** The Go `VulkanEncoder` already keeps encoder weights and intermediates resident, uploads mel once, and downloads the hidden state once. It has a fence per plan/layer; the decoder remains CPU. Removing just the two boundary copies is unlikely to supply the major improvement. Quantised kernels, attention and submission costs matter more.

The recent Nemotron small-query dense tiling is not automatically a Whisper encoder optimisation: Nemotron's streaming query batches are tiny; Whisper's encoder window has up to 1500 positions. Each shape needs its own benchmark and numerical gate.

## Recommended order

1. **Benchmark the existing whisper.cpp Q5 Vulkan path on this exact Decoder file**, in an isolated process after the current job drains. Keep four CPU threads, the same input, no model/default changes, and measure decode/ASR/diarization separately. Preserve the existing VAD timestamp correction. Compare its transcript against the same reference and sampled audio, not merely hashes.
2. **Compare VAD off/on and word timing off/on explicitly.** VAD and alignment offer different functionality; neither should be hidden in a speed claim. Do not reduce beams, language scope or timestamps silently.
3. **If native Go is required, profile warm CPU and resident Vulkan encoder paths**, separating frontend, encoder, cross-KV preparation, token generation and alignment. Reuse bounded workspaces and remove repeated preparations only where ownership/numerical tests permit them.
4. **Evaluate existing selective-Q8 Vulkan encoder candidates**, with current error/timestamp/token gates. Full Q8 placement is not assumed safe. Avoid altering the default merely to hit a timing target.
5. **Use a smaller/distilled model only as a separately named trade-off.** It can reduce compute, but multilingual behaviour and recognition quality may differ. Turbo already has only four decoder layers, so treating it as a full 32-layer decoder would overestimate that opportunity.

The fastest route to testing better transcription quality at similar or lower latency is the retained whisper.cpp backend. A native-Go optimisation campaign is a separate engineering choice. Neither requires restarting the old services or changing the current transcription service.

## SIMD priorities for native Go

The user explicitly requested SIMD where needed. Existing `Sdot`, `Sdotx4`, `Saxpy`, layer-normalisation and GEMM kernels already provide vector execution; a blanket SIMD rewrite would duplicate them.

1. **Fuse/batch decoder projection rows.** `model/whisper/decoder_bufs.go:linearInto` loops serially over output rows and calls `Sdot` for each. The separate allocating `linearForwardOpt` already has a parallel single-token path, but the buffer-based hot path does not use it. Profile fused multi-output GEMV and bounded row parallelism for 1280-wide projections and the vocabulary head. Preserve current reductions and caller-owned output.
2. **Fuse attention value accumulation.** `decoder.go` calls `Saxpy` once per cache row, both for self attention and head-major cross attention. A scoped multi-row value kernel could reduce call overhead. The new generic `AttentionValueRowTo` is a useful starting point, but Whisper skips weights below `1e-8` and may use a different fused-arithmetic contract. It cannot be substituted blindly. Preserve head layout, masking, skip rules and existing numerical behaviour.
3. **Share immutable alignment preparation.** Decoder scratch already exists, and cross-KV is already cached within a window. The second word-alignment decoder state repeats cross-KV work. A checked read-only sharing contract could remove that preparation while keeping mutable self-KV and observer state independent.
4. **Evaluate an explicit amd64 INT8 backend.** `model/whisper/int8_linear.go` is RISC-V-only; `int8_linear_other.go` sets `useInt8=false` and `int8Eligible=false` on Sigma. Setting `WHISPER_INT8` alone therefore does not enable x86 Whisper INT8. Generic Q8 SIMD helpers exist, but the inspected amd64 helper dequantises integer weights and uses floating-point FMA, rather than AVX-VNNI integer dot products. Wiring immutable packed weights or implementing a proper VNNI path requires separate calibration, finite/shape/alias gates and transcript/timestamp quality checks.
5. **Profile encoder tiles and scratch reuse separately.** Large Whisper encoder matrices do not share Nemotron's tiny-query shape. Avoid applying its four-row worker policy indiscriminately. Avoid nested worker oversubscription; test total request time at the four-thread budget.

Exact-order SIMD and allocation changes come first. Quantised weights/activations remain a separately named, opt-in numerical trade-off. No SIMD speed improvement is measured by this source review; native benchmarks must wait for the active Decoder pipeline to drain.

## Quality and current scope

The current Nemotron transcript has useful gist but errors in names and quotations. Whisper may improve those cases, but the earlier short multilingual corpus does not establish that it will do so on this interview. A same-file, same-resource comparison is needed.

The current Decoder run uses Vulkan zero-copy only at Nemotron's subsampling projection boundary, then CPU encoder/RNN-T and CPU diarization. It stays isolated on port 18094. No Whisper model was loaded and no competing inference was launched for this assessment. A read-only delegated source review timed out and supplied no independent findings.

Detailed historical methods and caveats: `projects/whisper-stt/docs/measurements.md` and `README.md`. Current Go source: `model/whisper/pcm_transcribe.go`, `encoder.go`, `decoder.go`, `linear_opt.go`, `word_alignment.go`, `vulkan_encoder.go`.

## Retained timing verification

The four Q5 timing records selected from `projects/whisper-stt/benchmarks/gate.json` were:

| Record ID | Seconds |
|---|---:|
| `20-vq5-pt-real-2` | 6.690557468 |
| `23-vq5-pt-real-2` | 6.685063928 |
| `25-vq5-pt-real-2` | 6.675808757 |
| `26-vq5-pt-real-2` | 6.677242931 |

Their mean is 6.682168271 seconds; division by the fixture's 43.76 seconds gives RTF 0.152700372. The cited `projects/whisper-stt/` files are retained workspace evidence outside this repository; private media, model weights, runtimes and job databases are not published with this assessment. This commit changes documentation only.
