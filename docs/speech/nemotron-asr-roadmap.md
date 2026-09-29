# NVIDIA Nemotron ASR roadmap

Go-pherence runs the pinned Nemotron 3.5 ASR streaming checkpoint through a bounded native PCM-to-RNN-T-decisions path with a 24-layer cached encoder and text decoding. SIMD has model-oracle parity on JFK, tiled JFK and a podcast crop. PTX and Vulkan can optionally run one subsampling projection inside the CPU request. Neither GPU path runs the full encoder and decoder. The [bounded reference record](../validation/nemotron-speech-reference-2026-09-28.md) identifies the released checkpoint and processor.

## Hybrid GPU projection requests

`PCMGenerationStream.Projector` optionally selects a per-request
`DeviceSubsamplingProjector{Backend: "ptx"}` or `"vulkan"`. The projector
keeps the released `[1024,4352]` subsampling weight and four-row device
buffers resident until `Close`. Every unmasked generation chunk uploads its
four-row input, dispatches the projection and downloads owned output.
The frontend, causal convolutions, 24-layer cached encoder and RNN-T
prediction still run on CPU. The caller must close the projector on success
or failure; the opt-in parity test does so even after an early failure.
GPU errors close the PCM stream without committing a failed subsampling
cache transition. The diarisation GPU path has the same limited scope.

On the i7-12700/RTX 3060 at `GOMAXPROCS=4`, full-request elapsed time
(including projector preparation, transfers, dispatches and teardown;
excluding checkpoint load) for 11-second JFK was `8.09 s` on SIMD,
`8.33 s` with PTX and `9.48 s` with Vulkan. Both hybrid paths matched
all 185 reference decisions, 45 nonblank emissions, absolute frames and
text with zero subsampling/tower outliers under the existing gates.
On a 20-second podcast crop, the single samples were `15.86 s` SIMD,
`15.58 s` PTX and `15.80 s` Vulkan; both hybrid paths matched 419
decisions, 167 nonblank emissions, frames and text. The two composed
PCM-to-subsampling outliers persist on both hybrid paths and are
unqualified. Both GPU paths also matched all 1,663 post-BOS decisions,
frames and text on the 100-second tiled JFK reference (`74.72 s` PTX,
`74.80 s` Vulkan, single samples). Dispatch-count and resource-close
assertions guard the tests. There is no demonstrated whole-request GPU
speedup, full model residency, labelled WER or sustained throughput.
Run the opt-in parity test with
`GO_PHERENCE_NEMOTRON_ASR_GENERATION_PROJECTOR=ptx` or `vulkan` and the
existing model and matching generation reference variables.

## Pinned candidates

- [Nemotron 3.5 ASR Streaming 0.6B](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/tree/ea30d66debe3740a08b573244286791d423d6b3e), revision `ea30d66debe3740a08b573244286791d423d6b3e`. The pinned [model card](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/blob/ea30d66debe3740a08b573244286791d423d6b3e/README.md) describes a multilingual, cache-aware FastConformer/RNN-T with language-ID prompting, configurable streaming chunks, punctuation and capitalisation. It claims 40 language-locales, of which 32 are transcription-ready or broad-coverage. The inspected `config.json` names `Nemotron3_5AsrForRNNT`, 24 encoder layers and 128 mel bins. The model card SHA-256 is `a3344caadf796c084c6b90a9fa5978068fd45e3a019790bebe50489bb3c0f7b7`; config SHA-256 is `62d186fd91f518e00e7867500f1f5819225e8ee95ea3e21b546514bf2048e845`. Hub licence metadata links to OpenMDW-1.1.
- [Nemotron Speech Streaming EN 0.6B](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b/tree/ebe59e5a817142986528bbbee5dba8db7b38ed50), revision `ebe59e5a817142986528bbbee5dba8db7b38ed50`. The English-only model card describes a cache-aware FastConformer/RNN-T, and its config names `NemotronAsrStreamingForRNNT`. Its card SHA-256 is `7701f4b2f1c16542c8ceb2b3a61dd144032898c17f4dc9cc1bbecda6972edec8`; config SHA-256 is `dffe850bc79ad2b0f8117804502b24d2c4a445aafbed4c1e40f8d78e0cb44065`. Hub licence metadata names the NVIDIA Open Model License. Review each model's actual licence terms and artifact rights before use.

These are distinct checkpoints and licences. Their cards use Parakeet-related architecture tags, but the [Parakeet TDT v3 roadmap](parakeet-asr-roadmap.md) targets a separate TDT decoder. [Nemotron 3 Diarization](nemotron-3-diarization-roadmap.md) supplies speaker attribution, while [NemotronLabs VoiceChat](nemotron-voice-roadmap.md) includes speech output; neither is an ASR oracle for these checkpoints.

## Gates

1. The 3.5 ASR revision is selected for this slice, with processor, tokenizer and safetensors payload staged after user approval. Inspect the model-header inventory, frontend and chunk/cache contract. Freeze exact language/locale IDs, prompt and timestamp conventions and size limits.
2. Generate independent NeMo/Transformers fixtures for frontend, subsampling, cached attention/convolution state, RNN-T predictor/joiner and decoding. Test cold and warm chunks, different lookaheads, silence, language selection, cancellation and restart. Pin input audio hashes, token/text outputs, units and numerical tolerances.
3. Implement a scalar Go reference with owned bounded stream state, offline and incremental paths, deterministic overlap/timestamp handling and memory admission. Qualify parity before moving measured kernels to checked SIMD or NVIDIA backends.
4. Add explicit ASR provider selection to the [speech workflow](speech-integration.md). Compare accuracy on held-out labelled speech, language-specific word error rate, latency, throughput and recovery after cancellation. Preserve the existing Whisper/MOSS and optional speaker-provider contracts.

The shared `loader/audio/NemotronMelStream` frontend now accepts successive mono 16-kHz PCM chunks of up to five seconds, holding only a 512-sample pre-emphasis ring and fixed STFT/FFT scratch. It returns owned finalized feature rows and a masked zero row on `Finish`, independent of recording duration. Chunked 11-second JFK and 100-second tiled JFK tests match independent pinned Transformers features on SIMD and `GODEBUG=cpu.all=off`: the 100-second test returns 10,001 rows, maximum absolute error `3.67e-4`, mean `1.15e-6`, zero values outside `5e-4 + 1e-5*abs(reference)`. A one-CPU five-run 100-second frontend benchmark (five-second calls, excludes WAV loading) initially measured `80.7–101.7 ms`, `10,382,336 B/op` and `10,022 allocations/op`. Writing each feature row directly into the owned returned chunk removed per-row temporary allocations; five matching 3-iteration samples then measured `76.7–77.2 ms`, `5,262,336 B/op` and `22 allocations/op`. The short sample sets do not establish a sustained speedup. These timings measure frontend processing, not a full ASR request or concurrent service. The ASR model still needs streaming subsampling, encoder-cache scheduling, masks, incremental RNNT state and text assembly before arbitrary-duration transcription. The 32-frame CPU `model/nemotronasr/Subsampling.ForwardOffline` path covers the released stem, two depthwise-separable stages and 4352→1024 projection. Independent PyTorch fixtures cover the activated stages and output at valid lengths 32, 31, 16, 1 and 0. The offline length rule at every stride is `floor(valid/2)+1`; the final projected output at valid=32 had maximum absolute error 0.000641 and mean absolute error 0.0000523 on AVX2/FMA. A one-CPU 32-frame microbenchmark after removing the depthwise transpose yielded 4.63–6.35 ms/request, 2,494,464 B and 10 allocations (five 20-iteration runs; includes owned output and intermediates, excludes loading and frontend). The prior path used 2,895,872 B and 12 allocations under the same bench settings; timing samples overlap, so there is no speed claim. The first ASR encoder block's initial 1024→4096→1024 SiLU feed-forward and half-scaled residual now pass independent five-row PyTorch stage fixtures. With the exact projected fixture as input, the residual had max error `0.000977`, mean `5.60e-6`, and no values outside `3e-4 + 2e-5*abs(reference)` on AVX2/FMA. Composing the Go subsampling path raises the mean to `5.92e-5`; two near-zero outputs exceed that standalone gate. Independent PyTorch FF1 on Go's subsampling output reproduces those two outliers, so the composed gate is separately bounded at `1e-3 + 2e-5*abs(reference)` per value and `1e-4` mean. The scalar CPU fallback also passes. Five 20-iteration, one-CPU isolated FF1 benchmarks ranged 2.01–3.64 ms with 122,880 B/3 allocations per five-row call; loading and subsampling are excluded. The first encoder attention's pre-normalisation and bias-free Q/K/V projections also pass five-row PyTorch fixtures from the FF1 residual: AVX2/FMA maximum errors `9.54e-7` (Q), `2.38e-6` (K), and `1.43e-6` (V), with zero outliers. A one-CPU five-row Q/K/V benchmark ranged 0.624–0.773 ms, 81,920 B and four allocations across five 20-iteration runs. The first encoder attention's complete unmasked five-row window also passes independent PyTorch positional encoding, Transformer-XL relative shift, Q/K/V, softmax and output/residual fixtures: AVX2/FMA maximum error `0.000145`, mean `1.14e-5` for the residual, with zero outliers. The scalar fallback passes the same gate. Five 20-iteration one-CPU attention samples took 1.70–1.79 ms, 225,280 B and nine allocations per five-row window, excluding subsampling and FF1. The first encoder block's convolution module now passes independent five-row PyTorch pre-norm, pointwise, GLU, causal nine-tap depthwise, post-depthwise norm/SiLU, output and residual fixtures. On AVX2/FMA the output and residual maximum error is `0.000488`, mean below `7e-6`, and zero values outside the numerical gate; scalar fallback passes. The isolated one-CPU convolution benchmark ranged 0.697–0.865 ms with 163,840 B/seven allocations across five 20-iteration runs. The full first encoder block now includes FF2 with its half-scaled residual and output LayerNorm. Five-row PyTorch stage fixtures pass for FF2; the output-normalised full block has maximum absolute error `3.05e-5` from the independent projected fixture and `6.25e-5` when composed with Go subsampling on AVX2/FMA, both with zero outliers. Scalar fallback also passes. Five one-CPU 20-iteration block samples took 7.26–7.44 ms and 655,360 B/23 allocations per five-row call, excluding frontend, subsampling and loading. The earlier five-row attention/block fixtures call the reference attention operator directly with no mask. The encoder's actual `and_mask_function` intersects the bidirectional mask with a chunk rule: at default lookahead three, rows 0–3 cannot attend row 4; at zero lookahead the five-row mask is triangular. Independent pinned PyTorch mask and attention fixtures now qualify both five-row modes through `Encoder0Attention.ForwardOfflineLookahead`: AVX2/FMA residual maximum error `0.000244` for zero lookahead and `0.000145` for three lookahead, mean below `1.3e-5`, zero outliers; scalar fallback passes. The full five-row block now also passes independently generated zero- and three-lookahead masked PyTorch fixtures via `Encoder0Block.ForwardOfflineLookahead`: SIMD maximum absolute errors `6.10e-5` and `2.29e-5`, mean below `2e-6`, zero outliers; scalar fallback passes. The original unmasked `ForwardOffline` fixture remains a distinct operator check. A separate first-block causal-convolution padding cache now matches independent PyTorch padded-input, eight-frame state and depthwise output fixtures for three prepared-GLU chunks of 1, 2 and 2 frames. State and padded inputs agree numerically with the fixture; depthwise maximum error is `3.81e-6`, with zero outliers. The state is owned per stream and validation errors preserve it. Five 20-iteration one-CPU three-chunk samples took 0.106–0.157 ms and 241,664 B/nine allocations; this excludes GLU, attention and the rest of the block. The first attention layer's sliding key/value cache now passes independent PyTorch returned/state K/V fixtures for three prepared projected chunks of 1, 2 and 2 frames: maximum error `2.38e-6`, zero outliers. A separate 40/20/7-frame overflow test checks the pinned `DynamicSlidingWindowLayer` rule: return 40/60/63 visible frames, retain at most 56, track all 67 seen frames. Returned arrays and snapshots are owned; rejected updates preserve state. The cache append/retention fixture is separate from cached attention. `Encoder0Attention.ForwardCachedChunk` accepts at most five cumulative already-normalised rows and matches independent pinned PyTorch outputs for prepared chunks [0,1], [1,3] and [3,5] at lookahead zero and three. The fixture generator saves each position tensor, mask and output from a fresh `DynamicCache` per mode. AVX2/FMA maximum absolute error is `0.000290`, with mean up to `2.83e-5` and zero values outside `3e-4 + 2e-5*abs(reference)`; the scalar fallback also passes. The zero-lookahead middle chunk has the largest mean error, so there is no stricter mean-error gate. Regenerating the pinned fixtures kept masks and positions numerically identical; output changes up to `0.000137` reflect CPU numerical variation, and the existing per-value gate passes. The Transformers `image_like_kwargs` diagnostic appears on stderr, but the fixture process exits successfully and generates all six outputs. Cached attention commits its per-stream K/V state only after a finite output is produced; rejected-output tests verify state preservation. A separate `Encoder0Block.ForwardCachedChunk` now composes FF1, cached attention, causal convolution history and FF2 on those three prepared projected chunks for lookahead zero and three. The two per-stream caches commit together only after finite output; rejected chunks preserve both. Six independent PyTorch block-output fixtures pass AVX2/FMA with maximum absolute error `7.47e-5`, mean below `7.4e-6` and zero values outside `3e-4 + 2e-5*abs(reference)`; the CPU-features-disabled path also passes, with maximum below `8.40e-5`. This is first-block chunk-seam parity over five prepared subsampled rows. The Nemotron 3.5 prompt-fusion and RNN-T joint operators now have a separate released-weight check using the five-row first-block fixture as prepared input, prompt IDs 101 (default) and 7, and a fixed released token embedding as the decoder vector. `RNNTProjection.Project` checks 1152→2048→1024 prompt fusion and 1024→640 encoder projection; `Joint` checks ReLU and 640→13088 vocabulary logits. PyTorch stage fixtures pass both SIMD and CPU-features-disabled paths. The composed encoder projection has maximum errors `7.33e-4` (SIMD) and `2.69e-3` (scalar), mean below `1.3e-4`, and zero values outside a calibrated `8e-4 + 2e-5*abs(reference)` per-value gate. At the exact pinned prompt-output fixture, an independent F64 dot had zero outliers under the tighter `3e-4 + 2e-5*abs(reference)` gate; composed prompt rounding and reduction order explain the looser near-zero floor. The joint logits reach maximum error `0.086` on the scalar path at values up to roughly 82,000, with zero per-value outliers under `3e-4 + 2e-5*abs(reference)`. These inputs are not full-encoder output or an LSTM prediction, so they say nothing about decoded tokens or transcription quality. A separate single-stream `RNNTDecoder.Step` now matches five pinned PyTorch cached predictor steps for tokens `[blank, 3, 7, blank, 11]`. It loads the released 13,088×640 embedding, two 640-wide LSTM layers and 640-wide decoder projector. The initial blank runs the LSTM; the later blank retains output, hidden and cell state without mutation. AVX2/FMA output, hidden and cell maximum absolute errors are below `5.47e-6`, with zero values outside `3e-4 + 2e-5*abs(reference)`; CPU-features-disabled tests and malformed-cache checks pass. This does not compose real encoder frames with the joint network or implement greedy token selection, multi-batch state, or transcription. `LoadIndexedEncoderBlock` also loads the released layer-1 weights into the bounded block math. Independent full five-row PyTorch fixtures for lookahead zero and three check the layer-1 attention pre-normalisation and output composed after Go layer 0: AVX2/FMA output maximum absolute error `8.40e-5`, mean below `5.7e-6`, zero values outside `3e-4 + 2e-5*abs(reference)`; the CPU-features-disabled path also passes with maximum below `9.16e-5`. `OfflineEncoderTower.ForwardOfflineLookahead` now composes all 24 released encoder layers on the same five-row subsampled window. Sparse independent PyTorch checkpoints after layers 7, 15 and 23 pass both lookahead modes on AVX2/FMA and with CPU features disabled, with zero values outside `3e-4 + 2e-5*abs(reference)`. Maximum absolute error through layer 15 is `7.63e-5` on SIMD and `1.84e-4` on scalar; final layer-23 output maximum error is below `6e-8` on SIMD and `2e-7` on scalar. The final output's small magnitude follows the reference; this test does not measure WER or certify larger windows. `OfflineProjection.ForwardFeatures` now composes the released 32-frame subsampler, 24-layer five-row encoder and default-prompt (101) fusion/640-wide encoder projector. Independent PyTorch fixtures from the prepared JFK processor features pass both lookahead modes: AVX2/FMA maximum absolute error below `2.87e-6` for projected states, scalar below `4.30e-6`, with zero values outside `3e-4 + 2e-5*abs(reference)`. The final encoder input and output were independently regenerated and agree with the earlier five-row fixtures. A bounded single-stream `GreedyRNNT.Decode` now joins the released predictor cache, joint vocabulary head, blank-frame advance and ten-symbol frame limit for up to five prepared projected rows. Pinned PyTorch per-step logits for both lookahead modes pass SIMD and CPU-features-disabled checks with zero values outside `3e-4 + 2e-5*abs(reference)`; maximum errors are `7.33e-4` and `3.18e-3`, respectively. Both initial five-frame fixtures select blank at every frame, with a minimum reference top-two logit gap above `7.4`; a separate forced-nonblank transition test checks the ten-symbol limit. A further independent PyTorch fixture takes five projected encoder rows (positions 10–14) from the released model's full 11-second JFK inference with lookahead 3 and default prompt 101. Native greedy decoding matches all eight PyTorch token/frame decisions, including three nonblank emissions and subsequent blank transitions: tokens `[13087,13087,2860,13087,13087,2,1290,13087]` at frames `[0,1,2,2,3,4,4,4]`. Per-step joint logits have maximum absolute error `3.06e-5` on AVX2/FMA and `1.53e-4` with CPU features disabled, zero per-value outliers under `3e-4 + 2e-5*abs(reference)`, and mean below `1.4e-5`. The same released PyTorch full-JFK encoder supplies 139 projected rows for a separate native greedy-decode test. The bounded `GreedyRNNT.Decode` path now accepts up to 139 already projected rows and matches all 187 PyTorch token/frame decisions, including 48 nonblank emissions. Pinned selected and runner-up token IDs/logits match on SIMD and CPU-features-disabled paths; the maximum selected-logit error is `2.45e-4` and `9.77e-4` respectively, with zero calibrated per-value outliers. The smallest reference top-two margin is `0.003885`, and at every step it exceeds the measured sum of selected and runner-up errors; no broad decision-stability claim follows beyond this recording. These encoder rows are **PyTorch-produced**. The native five-row and 139-row tests qualify the predictor/joint/token loop, not native full-recording encoder inference, text decoding, WER or transcription accuracy. The path returns raw tokens and frame indices, not text. The decoder now reuses its joint activation and vocabulary-logit scratch across decisions, while `RNNTProjection.Joint` and the optional per-step observer still return owned results. On a one-CPU 139-row projected JFK decode, three five-iteration runs measured 12,395,264 B/720 allocations before and 1,169,280 B/346 allocations after. The greedy loop also borrows the predictor's current state output internally, while public `RNNTDecoder.Step` still returns an owned copy. Another three five-iteration samples on the same one-CPU setup measured 666,624 B/159 allocations. Times for the baseline, first scratch change and borrowed-predictor change were 308–323 ms, 300–302 ms and 240–296 ms respectively; variability and overlapping ranges preclude a speed claim. The benchmark excludes checkpoint loading and all encoder/frontend work. Released PyTorch token, frame, selected/runner-up logit parity passes with SIMD and CPU features disabled. `OfflineProjection.ForwardPCM32` now accepts the first 5,160 samples of the pinned 16-kHz JFK waveform, computes 33 mel rows, drops the terminal masked row, and runs those 32 valid rows through the short-window encoder/projector. The extra 200 PCM samples supply frame 31's complete STFT context. Comparing against the independent full-recording PyTorch processor crop and model fixtures passes both lookahead modes: AVX2/FMA maximum projected-state error `2.99e-6`, CPU-features-disabled `4.18e-6`, zero per-value outliers. A 4,960-sample prefix failed this gate because it masked frame 31; it is not an equivalent crop. `PCM32RNNT.Decode` now composes that native fixed-prefix frontend, 24-layer encoder/projector and single-stream greedy decoder, returning owned raw token IDs and frame indices. On the same 5,160-sample JFK prefix, both lookahead 0 and 3 select five blanks at frames 0–4, matching the independently pinned PyTorch per-step vocabulary logits with zero per-value outliers under `3e-4 + 2e-5*abs(reference)`. SIMD mean absolute logit error is below `1.77e-4`, scalar below `4.33e-4`; maximum errors are `0.00171` and `0.00562` respectively at larger-magnitude logits. A one-CPU i7-12700 benchmark of that owned PCM-to-token call (`GOMAXPROCS=1`, five five-iteration samples, model and WAV loading excluded) measured `195–230 ms/op`, `18,458,368 B/op` and `575 allocs/op`. This is a fixed short prefix, not integrated streaming, nonblank native audio decoding or a text transcription. Larger windows and the 1101-frame recording still have no native encoder parity or latency evidence. Published accuracy and latency claims belong to NVIDIA's evaluations; native transcription quality and production readiness are untested.

`SubsamplingStream.ForwardChunk` adds per-stream padding caches for all three
causal Conv2D stages. It admits fully valid 8–128-row mel chunks in multiples
of eight, holds one time row per stage, and commits caches only after a finite
projection. An independent pinned PyTorch fixture runs the JFK processor's
first 128 mel rows through 8-, 16-, 32- and 64-row chunks. All 16 projected
rows match the released-weight reference on SIMD and `GODEBUG=cpu.all=off`
with zero values outside `3e-4 + 2e-5*abs(reference)`. SIMD mean absolute
errors span `4.71e-5–5.00e-5`; scalar means span `5.90e-5–7.10e-5`. This
qualifies full-valid subsampling cache transitions only.

`SubsamplingStream.ForwardMaskedChunk` now returns projected rows and their
valid count. It copies caller input, zeros masked mel tails and masks output
after each Conv2D stride (`floor(valid/2)` per stage). A 13,000-sample JFK
prefix supplies a 26-row first processor chunk (25 valid), a 32-row next
chunk, and a terminal 32-row chunk (24 valid, eight right-masked). The next
chunk repeats first-chunk row 25 as a valid mel row; the first chunk's row 25
is masked. The pinned processor independently produces the features using
`center=True` on the first 4,040 samples and `center=False` on subsequent
overlapping PCM windows. Its released-weight streaming subsampler produces
three four-row outputs, with valid counts 3/4/3. Native SIMD matches all
12,288 values with maximum absolute error `6.11e-4`, mean `4.18e-5`, zero
values outside `3e-4 + 2e-5*abs(reference)`; CPU features disabled matches
with maximum `1.22e-3`, mean `5.66e-5` and zero outliers. The first
chunk's 26th mel frame is masked and reappears as a valid frame in the next
chunk; the last subsampler convolution row is internally masked in the
first and terminal chunks. The encoder uses a separate output-mask formula:
the pinned reference marks all four projected rows visible for both 25/26
and 24/32 chunks. `EncoderMaskRows` records that rule; no full encoder
masked-row parity check has passed yet. These are subsampling-only checks. `ASRMelChunkStream` now schedules arbitrary PCM
calls into these first/subsequent/terminal shapes, holding fewer than 32
pending mel rows. An 11-second JFK check produces 35 owned chunks and 1,100
valid features with maximum error `3.66e-4` against the pinned frontend;
a 100-second synthetic check produces 313 chunks and 10,000 valid rows
without growing pending mel state. The composed 13,000-sample PCM→masked
subsampling comparison has 44 of 12,288 projected values outside
`8e-4 + 3e-5*abs(reference)` (maximum `0.00238`, mean `1.65e-4`). Pinned
PyTorch subsampling on the native mel rows has zero outliers under the
operator gate (maximum `6.11e-4`, mean `4.28e-5`), identifying frontend
rounding amplified by the released subsampler. This failed composed gate
is retained; it needs downstream decision/error analysis, not a tolerance
change alone.

The pinned cache-aware `generate()` path has a different input contract:
lookahead three requires **25** mel rows first and **32** thereafter; it
passes no attention mask to the encoder. It discards the first processor's
masked 26th row and pads the final chunk with zero mel rows. Passing a 26-row
first chunk raises `ValueError`. `SubsamplingStream.ForwardUnmaskedChunk`
implements that 25/32 cache schedule and rejects mixing it with masked
cache calls. A 13,000-sample JFK prefix, fed to the native mel scheduler in
397-sample PCM calls, passed three released-weight PyTorch generation-path
checkpoints: subsampling `[12,1024]`, layer zero, full 24-layer tower, and
prompt-projected `[12,640]` states. On SIMD, the largest absolute errors
were `0.00238` at the subsampling output, `2.37e-4` at layer zero,
`2.16e-7` at the tower and `4.41e-6` at the RNNT projector, with zero
per-value outliers under their respective calibrated stage gates. The
CPU-features-disabled run likewise had zero outliers (largest tower error
`3.28e-7`, RNNT error `7.87e-6`). Native incremental greedy decoding made
12 blank decisions at frames 0–11, matching the pinned reference
`generate()` after its initial BOS blank. This short prefix contains no
nonblank decision, so full-recording transcription and text/WER remain
unqualified. The 26/32 **masked** subsampler fixture above is a separate
operator probe; a direct masked encoder attempt diverged from step two
because its 2D chunk mask did not include cached history. No masked 24-layer
claim follows from that probe. Long audio, independent token/text accuracy,
WER, and transfer-inclusive GPU timings still need validation.

`PCMGenerationStream` now composes the default lookahead-three generation
schedule, unmasked cached subsampling, 24-layer cached tower, prompt 101
projection and incremental greedy RNN-T in one per-stream state. It accepts
arbitrary finite PCM calls up to 80,000 samples, checks cancellation between
model chunks, and returns owned raw decisions with absolute encoder frame
indices. Frontend, convolution, attention and predictor state stay bounded
independently of recording duration; the caller must consume returned
per-call decisions to avoid accumulating an unbounded recording in memory.
Invalid PCM leaves the stream unchanged; cancellation or a model failure
closes it. Three different PCM call sizes on the first 13,000 JFK samples
produce 12 blank decisions at frames 0–11, agreeing with pinned
`generate()` after its initial BOS blank. `DecodeRNNTText` uses the released
Parakeet vocabulary, removes blanks and special IDs, preserves repeated
RNN-T tokens (no CTC collapse), and yields an empty string on this prefix.
The 11-second JFK recording now passes an independent pinned Transformers
streaming `generate()` comparison at lookahead three. `AppendPCM` calls of
397, 4,040 and 5,520 samples each produce 185 matching decisions after
removing the reference's BOS blank, including 45 nonblank emissions, at
identical encoder frames. `DecodeRNNTText` returns the same text:
“And so my fellow Americans ask not what your country can do for you.  Ask
what you can do for your country”. The processor's 25/32 unmasked mel
schedule yields 140 encoder rows. SIMD subsampling output over these rows
has maximum absolute error `0.00238`, mean `1.26e-4`, zero outliers under
`3e-3 + 4e-5*abs(reference)`; the full 24-layer output has maximum
`3.16e-6`, mean `1.83e-8`, zero outliers under
`3e-4 + 2e-5*abs(reference)`. The original step-15 run failed: native
attention differed by up to 16.22 and the token loop first differed at
frame 72. Correcting sliding relative positions to the visible K/V length
removed that error. A CPU-features-disabled full-JFK run exceeded the
five-minute command limit after reaching chunk 21; scalar full-recording
parity is unverified. This is one recording's transcript parity against
a model oracle, not measured WER on labelled speech, a quality comparison
across accents or production latency. A single CPU-features-disabled run
with 5,520-sample PCM calls subsequently completed in 80.13 seconds:
185 decisions, 45 nonblank emissions, frames and text matched PyTorch;
subsampling maximum/mean errors were `0.00238`/`1.37e-4`, and tower
`4.08e-6`/`2.82e-8`, with zero calibrated outliers. The earlier
three-call-size scalar run exceeded five minutes; that timeout was not a
numerical failure.

A `GOMAXPROCS=4` i7-12700 profile of the complete 11-second JFK PCM→token
benchmark measured `7.79 s/op` over two requests and `1.53 GB/op` after
loading. The sampled CPU time was dominated by the SIMD FMA SGEMM tile
(`19.15 s` flat of `26.92 s` sampled across benchmark and setup); the
allocation profile also includes model loading, so its `GetFloat32` share
is not a request-allocation estimate. Bounded five-row ASR matrix trials
found prepacking alone slower than the existing blocked path: FF1 4096×1024
`2.81–3.15 ms` versus `0.96–1.00 ms`, FF2 1024×4096 `2.53–2.92 ms`
versus `0.98–1.01 ms` (`GOMAXPROCS=4`, two 15-iteration samples).
A four-worker pooled panel run overlapped the blocked timings, so no
ASR runtime dispatch change or request speedup follows from these isolated
trials. Results from longer-row diarization GEMMs do not transfer to the
five-row ASR encoder contract.

The cached encoder now retains each four-query-row relative-position
projection once per model and visible K/V length; other chunk sizes keep
their original exact-length projection. Each cached tensor is immutable
after `sync.Once` publication, so independent streams can share the model.
On the same i7-12700 at `GOMAXPROCS=4`, three complete JFK benchmark samples
fell from `7.55–7.70 s/op`, `1.529 GB/op`, 28,884 allocations to
`6.29–6.41 s/op`, `1.368 GB/op`, 28,164 allocations (three requests per
sample, model/WAV load excluded). The 72-row cached tower (lookaheads zero
and three), 11-second PCM→text (three chunk sizes), and 100-second tiled JFK
passed their independent reference gates with zero token mismatches and
zero tower outliers. A shared-model two-stream `-race` run passed.

Both ASR feed-forward modules now reuse their owned normalisation buffer
as the fc2 GEMM destination once fc1 has consumed it. The accumulating
GEMM clears that destination; the caller's input and FF2's owned final
output remain independent. Two complete 11-second JFK request samples
allocated `1.349 GB/op`, 26,544 allocations versus `1.368 GB/op`, 28,164
allocations before reuse (`GOMAXPROCS=4`, model/WAV load excluded); timings
overlap, so no speed claim follows. Pinned FF1/FF2, both 72-row cached tower
lookaheads, 11-second PCM→text (three chunk sizes) and 100-second tiled JFK
pass with zero tower outliers and matching token decisions. The shared-model
two-stream `-race` test passes on this change.
These are single-host samples; labelled WER and broader quality are open.

A one-CPU i7-12700 full-request benchmark (11-second JFK, five-second
`AppendPCM` calls, model/WAV loading excluded) identified relative-position
encoding and projection in the cached encoder as repeat costs. Computing
the 512 inverse frequencies once per encoding and projecting only the
`keyRows + rows - 1` positions consumed by `_rel_shift` preserved pinned
11-second PCM→text, regenerated 72-row cached tower/attention, and shorter
operator parity. Three two-iteration samples before narrowing the projection
measured `10.12–10.50 s/op`, about `2.011 GB/op` and 29,724 allocations;
three matching samples after measured `8.38–8.94 s/op`, about
`1.859 GB/op` and 29,724 allocations. These small samples indicate a local
CPU reduction, not sustained or GPU speed. A one-iteration scalar full
request measured 104.02 seconds and 2.010 GB; it is not paired with a
scalar baseline. Returned token/logit slices and all model state retain
their existing ownership and bounded-memory contracts.

A 100-second tiled-JFK reference now drives the pinned cache-aware
`generate()` path using 313 mel chunks (25 first, then 32). The native
`PCMGenerationStream` consumes twenty five-second PCM calls, returns 1,252
encoder frames, and matches all **1,663 decisions after BOS**, including
411 nonblank emissions, at identical frame indices. `DecodeRNNTText`
matches the pinned 973-character output. The full native run completed in
85.84 seconds on the i7-12700 with `GOMAXPROCS=1`; model loading and
reference generation are excluded. Across 1,252 encoder rows, subsampling
had maximum/mean absolute errors `0.00322`/`1.28e-4` with zero outliers
under `3e-3 + 4e-5*abs(reference)`. The full 24-layer tower had
`1.59e-5`/`1.80e-8` and zero outliers under
`3e-4 + 2e-5*abs(reference)`. The comparison uses a model oracle on
a tiled single-speaker recording; it is not labelled WER, varied speech
quality, cancellation soak, or GPU timing. The 11-second reference still
passes with 5,520-sample calls after the fixture extension.

A separate **real 20-second podcast crop** (300–320 seconds of the tracked
mono 16-kHz `testdata/podcast.wav`) now exercises different speech and
background. The pinned processor and released streaming `generate()`
produce 63 mel chunks and 420 decisions including BOS; native five-second
PCM calls match all **419 decisions after BOS**, 167 nonblank emissions,
absolute frames and the same 351-character text. The full tower has
maximum/mean absolute error `4.25e-6`/`1.49e-8`, zero values outside
`3e-4 + 2e-5*abs(reference)`. The composed PCM→subsampling comparison has
**two of 258,048** values outside its provisional
`3e-3 + 4e-5*abs(reference)` gate, both in chunk 36 (maximum overall
`0.00458`, mean `9.13e-5`). Those failures remain recorded; matching
downstream tokens does not qualify the subsampling stage or measured WER.
An isolated operator check now feeds the pinned 2,000 podcast mel rows
into native unmasked subsampling, retaining its convolution caches. Its
258,048 projected values have maximum/mean absolute error
`5.49e-4`/`4.11e-5` and **zero outliers under the same projection gate**.
The native PCM→mel stage has maximum/mean error `3.54e-4`/`8.40e-7`,
zero outliers under its separate `5e-4 + 1e-5*abs(reference)` gate.
Together these locate the two composed-stage outliers in frontend error
propagation, without attributing them to a failed isolated subsampling
operator or widening either threshold. They remain unqualified at the
composed stage. A paired diagnostic now drives native PCM mel and pinned
PyTorch mel through separate native subsampling streams on that same crop.
The pinned-mel stream has zero projection outliers; the native-mel stream
reproduces the two composed outliers at chunk 36, row 1, columns 466 and
639. At those columns, substituting native mel changes the outputs by
`-0.003515` and `+0.003374` relative to pinned mel. Chunk-36 mel has
maximum/mean absolute differences `3.54e-4`/`4.86e-6`, with its maximum
at row 9, column 10. This isolates frontend error propagation without
identifying a safe numerical fix or changing the acceptance gate. Two
bounded frontend power trials—direct `re²+im²` and a float64 squared sum
before the magnitude cast—and scalar rather than SIMD per-frame mel
projection still produced two composed-stage outliers and were reverted.
The source WAV digest guards fixture provenance, not
generated-output acceptance. Transcript agreement on this clip is still
model-oracle parity, not a human-labelled accuracy result.

`Encoder0Attention.ForwardCachedChunk` now uses the visible K/V length for
relative positions after the 57-frame window starts sliding, and applies
the reference's chunk-distance mask. The prior prepared-row fixtures passed
cumulative `get_seq_length()` to the positional encoder and therefore missed
a live `encoder.forward` divergence at chunk 15. Regenerated independent
attention and 24-layer tower fixtures use `get_mask_sizes(chunk, 0)[0] - chunk`
as `cached_frames`. Both 72-row lookahead modes pass again with zero outliers;
layer-0 attention maximum error on SIMD is `1.91e-4` and the prepared tower
maximum step error in the regenerated run is `2.85e-6`. The old fixture
contract is superseded. A 72-row PyTorch fixture reuses
released-weight JFK projected rows, then normalises them with released
layer-0 weights; it scores eighteen four-row chunks at lookahead 0 and 3.
The Go path matches those outputs on SIMD (maximum `1.91e-4`, mean
`1.42e-5–1.46e-5`) and scalar (maximum `2.90e-4`, mean
`2.19e-5–2.21e-5`) with zero per-value outliers under
`3e-4 + 2e-5*abs(reference)`. A synthetic high-amplitude probe did exceed
that gate, so this evidence is bounded to released-audio-derived inputs.
The existing five-row attention and cached-block tests still pass. The
standalone 72-row attention test qualifies released-audio-derived cache
transitions; the full-generation check below covers the 24-layer composition.

`CachedEncoderTower` now advances 24 per-layer attention and convolution
caches together for one to five **prepared**, fully valid projected rows.
The pinned PyTorch fixture replays four JFK-derived subsampling rows through
eighteen four-row steps (72 rows), using the released model and lookahead 0
or 3. All 36 SIMD and scalar steps match PyTorch under
`3e-4 + 2e-5*abs(reference)` with zero outliers; the largest step error
observed was `8.26e-6` on SIMD and `6.62e-6` on scalar. A rejected input
or missing final layer leaves prior cache state unchanged. These repeated
prepared rows qualify the 24-layer cache transition and 57-frame sliding
boundary, not varied full-recording PCM or native RNN-T transcription.
An 11-second full JFK streaming-generation fixture now checks this cache
boundary on native PCM. Cached attention now shares immutable relative-position
encodings indexed by the visible K/V row count across layers and streams;
`sync.Once` computes each bounded table only once. On the i7-12700 with
`GOMAXPROCS=4`, three complete 11-second request samples fell from
`8.51–8.52 s` and `1.86 GB/op` to `7.46–7.76 s` and `1.53 GB/op`
(excluding model load). Fresh independently generated 72-row attention
and tower references, both lookahead modes, the complete 100-second JFK
PCM→text reference (1,663 decisions after BOS), and the real podcast
PCM→text reference (419 decisions) passed. Earlier 72-row fixture files
failed at the sliding boundary on both the unchanged code and this trial;
they were superseded only after regenerating with the pinned processor's
`get_mask_sizes(chunk, 0)[0] - chunk` visible cache length. The two
podcast PCM→subsampling outliers remain unqualified. These are small
single-host samples, not sustained speed or GPU timings. An opt-in
shared-model `-race` test runs two independent five-second JFK PCM streams
concurrently with 4,040- and 80,000-sample calls. Both reproduce their
serial 80-decision outputs and absolute frames, and previously returned
decisions remain unchanged. It checks shared immutable weights and the
relative-position table, not sustained multi-request throughput or a
cancellation soak. Labelled WER and full-request GPU timing still need
validation.

`GreedyRNNTStream.Append` now carries the released two-layer predictor and
joint scratch between fully valid, prompt-projected encoder chunks. It returns
raw token IDs, including blanks, with absolute frame indices. Against the
existing independently generated 139-row PyTorch JFK encoder fixture, chunk
sizes 1, 4, 17, 56 and 139 all select the same 187 decisions and 48
nonblank emissions at the same frames. Malformed inputs and frame-count
overflow leave predictor state unchanged; a mid-chunk model failure closes
the stream. This test isolates incremental RNN-T on **PyTorch-produced encoder rows**.
The separate native full-JFK result below checks PCM-to-text.
