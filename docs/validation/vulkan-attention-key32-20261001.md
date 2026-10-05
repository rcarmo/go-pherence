# Native Whisper key32 attention and gap-preserving VAD — 1 October 2026

The explicit F32 key32 attention candidate reduces measured Go Whisper request latency by a further 4.9–5.4% on four retained English/Portuguese/French fixtures. Original-gap-preserving native VAD completes seven bounded fixture arms with word timestamps. The original-engine speed target, independent word-timing accuracy and long-form quality are incomplete.

## Attention

[Key32 shader](../../backends/vulkan/shaders/attention_f32_key32.glsl) retains 16 queries per workgroup and expands the key tile from 16 to 32. Lanes compute two logits, evaluate probabilities in parallel and retain four output channels. Shared K/V storage is reused behind barriers. Increasing-key probability summation and online max/sum rescaling avoid a quadratic score allocation. Explicit F32 FMA changes dot/value rounding relative to the baseline; precision remains F32.

`NewVkAttentionKey32F32` and `NewVulkanEncoderTile64Key32` select this mode explicitly. Baseline attention, the default encoder, production profiles and services are unchanged. The candidate requires 14,528 bytes of shared memory. Existing non-causal/equal-head/shape/extent/alias/device-limit contracts apply. Masked or causal decoder attention is unsupported.

Shuffled scalar tile tests independently check shared and probability ownership, barrier phases, output tails, rescaling and a full-score F64 reference. Ordinary absolute/relative limits remain `2e-5`/`2e-5`; the existing high-logit stress budget remains `2e-4`/`2e-5`. Native analytic tests, guard buffers and complete synthetic encoder checks pass. Mid-forward cancellation, explicit native drain and reuse pass without CPU fallback. Full `[1500,20,64]` attention outputs differ from baseline by at most `2.79e-8` on the retained analytic input. Six timings per kernel in alternating ABBA blocks give about 89.8 ms baseline versus 78.4 ms candidate.

Larger 64-key and 32-query variants were slower and rejected. Key16 parallel probability evaluation improved kernel time about 3%; key32 parallel/FMA gave the larger improvement. An alternative grouped-dot reduction passed numerics but was slightly slower than FMA. External SPIR-V was used only in removed diagnostic tests; no production API reads external shader files.

## Repeated trained requests

Five samples per arm, explicit language, greedy checked decoding, same pinned model/audio, four threads and affinity CPUs 0–7, Iris Xe RPL-P/Mesa 26.1.5, isolated 8 GiB/no-process-swap container. Live Qwen idle and host available memory ≥6 GiB were checked throughout. Timings exclude model loading/preparation, which remain separately recorded. Sequential arms and later controls corroborate the gain; these are not fully interleaved cold-start distributions.

| Fixture, VAD/words off | Tile64/key16 median | Tile64/key32 median | Reduction |
|---|---:|---:|---:|
| JFK | 7.835 s | 7.449 s | 4.93% |
| Portuguese 0 | 7.789 s | 7.405 s | 4.94% |
| Portuguese 1 | 7.361 s | 6.971 s | 5.31% |
| French 0 | 7.244 s | 6.856 s | 5.35% |

Every candidate sample matches baseline tokens, text, segment times and window metadata. Baseline fixture medians are retained in the [tile64 qualification](vulkan-linear-regtile64-20261001.md). Combined JFK improvement from original Go Vulkan F32 request median 9.601 s is about 22.4%. Retained original whisper.cpp Vulkan process medians remain 6.183 s F16 and 3.350 s Q5 with different precision/loading boundaries; the target is unmet.

## Native VAD with real internal silence

The experimental [PCM VAD API](../../model/whisper/pcm_vad.go) adds explicit `PCMVADOptions.PreserveWindowGaps`. It greedily groups chronological eligibility spans when their full original-audio extent fits a Whisper window. It preserves the actual internal silence and decodes disjoint groups separately. Long contiguous speech still uses the normal checked window planner. There is no hidden retry from compact mode, interpolation or automatic quality-changing selection.

`OriginalAudio` records the continuous original audio retained per window; `OriginalSpeech` records its intersection with Silero speech eligibility. `CompactedWindow` describes a single concatenated retained-audio timeline across groups, including preserved internal silence. Segment/word times use original samples. Words may cover actual silence supplied to alignment; they cannot straddle deleted audio between independently decoded groups. The existing compact-mode crossing guard is unchanged. Resume identity, serving defaults and cross-window text reconciliation are unsupported.

Seven fixture arms passed five repetitions each with VAD and word timestamps: JFK, two Portuguese clips, French, JFK padded by 15 seconds at each end, complete silence, and two JFK copies separated by 33 seconds of silence. The last fixture yields two original-audio groups, 44 words and global retained-window coordinates meeting at sample 164800. All repeated outputs are identical, bounded and chronological. Silence emits no transcript. The four natural-language clips retain their expected text; natural overlap and independently listened/annotated timing are untested.

Matched JFK gap-preserving words-on comparison with `GOMEMLIMIT=4GiB`: 8.930 s key16 versus 8.544 s key32 (4.32%), exact span/word/segment/token output agreement. Padded JFK retains 22 words; word times shifted from the unpadded recording differ from an exact 15-second translation by up to 52 ms because leading silence changes the VAD recurrent context and detected crop. No tolerance was widened to turn this into exact timing parity.

The two-group repeated arm initially hit the unchanged 8 GiB container limit on its fifth request with default Go GC. An explicit `GOMEMLIMIT=4GiB` rerun completes all five requests (median 17.238 s), no OOM, peak observed container memory about 6.84 GB. Go's soft heap limit excludes mapped Vulkan and other native memory; it is separate from the hard container cap. Original raw failure/state and rerun evidence are retained. No resource-cap increase was used. Callers must configure a suitable heap budget for repeated/long requests; the library changes no global GC setting.

One original Q5 Vulkan/Silero/DTW smoke succeeded, but uses compacted speech with overlap and different segment boundaries. Its output does not qualify gap-preserving schedule equivalence, word alignment or speed acceptance. Different timestamp algorithms need an independent annotated reference.

## Gates and evidence

[Manifest and raw reports](../../benchmarks/speech-foundations/vulkan-attention-key32-20261001/manifest.json) retain pinned inputs, outputs, native checks, hard-cap OOM and bounded rerun evidence. Model-free tests run ten times; build/vet, whole-tree tests/race, ARM64/RISC-V cross-builds, layout/docs and all 26 shader validation/rebuild checks pass. These qualify the explicit bounded candidate and integration behaviour. Full original-matched precision/loading, acoustic/timing accuracy, long-recording completion and production deployment require further work.

[Native Whisper performance contract](../speech/whisper-go-performance-contract-20260930.md) · [Independent Silero graph qualification](native-silero-parity-20261001.md).
