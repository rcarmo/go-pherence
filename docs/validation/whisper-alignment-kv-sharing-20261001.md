# Whisper alignment cross-KV sharing — 1 October 2026

Reusing immutable CPU cross-attention K/V for word alignment saves about 123 MB per aligned window and reduces measured words-on request medians by 1.9–2.5% on four bounded arms. Word, token, segment, VAD-span and original-timeline outputs match the prior implementation exactly.

## State ownership

The checked PCM path creates a fresh decoder state for generation, then formerly recomputed identical cross-K/V and head-major copies for a separate alignment state. Its private `newAlignmentDecoderStateContext` now shares those four immutable arrays per layer. Outer slice headers are copied. Position, last token, self-K/V capacity, observer and scratch are freshly allocated; alignment cannot modify generation's self state.

This is a private same-window seam under the checked caller's exclusive-use contract. Public decoder-state construction and direct `AlignWordsChecked` contracts are unchanged. No GPU allocation, lease or externally mutable cross-KV is shared. Geometry, cancellation and CPU-only ownership are checked before return; failed construction returns no partial state. Callers must keep the immutable source cross-KV unchanged throughout both states' lifetimes.

Model-free tests compare logits and observed cross-attention probabilities bit-for-bit against independently recomputed state. They verify backing identity, separate self caches and scratch, outer-slice independence, unchanged shared values, every deterministic cancellation checkpoint, bad geometry and reduced allocation count. Ordinary tests run ten repetitions.

## Measurements

Five request samples each, Intel Iris Xe RPL-P/Mesa 26.1.5, explicit F32 tile64/key32, native Silero `PreserveWindowGaps`, word timestamps enabled, four CPU threads/affinity 0–7, isolated 8 GiB hard cap/no process swap, `GOMEMLIMIT=4GiB`. Live Qwen idle and host available memory ≥6 GiB are guarded. Loading/preparation are excluded from request medians and recorded separately. Arms are sequential rather than fully interleaved.

| Fixture | Before | Shared cross-KV | Reduction | Allocation saved per request |
|---|---:|---:|---:|---:|
| JFK | 8.544 s | 8.355 s | 2.22% | 122,954,552 B |
| Portuguese 0 | 8.495 s | 8.301 s | 2.28% | 122,966,008 B |
| French 0 | 7.479 s | 7.292 s | 2.49% | 122,949,144 B |
| Two JFK groups across 33-second silence | 17.238 s | 16.904 s | 1.94% | 245,879,488 B |

Savings use the second sample's total allocated bytes; small runtime variations accompany the fixed cross-KV footprint. All five outputs per fixture exactly match the baseline's segments, words, original speech/audio spans, window coordinates and tokens. The two-group arm keeps 44 words and global retained-window identity. No recognition or timestamp setting was removed.

[Raw reports, hashes and gate state](../../benchmarks/speech-foundations/whisper-alignment-kv-sharing-20261001/manifest.json) reference baseline reports in the [key32/gap-preserving campaign](vulkan-attention-key32-20261001.md). Full-tree tests/vet/build/race, model layout, doc links and ARM64/RISC-V cross-builds pass in isolation. Production and model/profile defaults are unchanged.

## Remaining target

The original-engine speed goal is incomplete. Retained original Vulkan Q5 and F16 timings use different precision/loading boundaries; independent acoustic and word-timing accuracy, finite VAD speech splitting, resume identity and long-recording completion remain unqualified.

Follow-up shader experiments preserved output but did not qualify a further whole-request improvement: packed F16 tile64 with checked scalar unpack was 19–37% slower on three projection shapes; a 128-column tile was 16–19% slower. Explicitly unrolled tile64 was mixed (about +1.6%, −3.4%, −5.0% kernel time); attention unrolling/changing dot reduction yielded only small kernel gains and has no trained whole-request acceptance. These diagnostic shaders were not adopted and their external-shader tests were removed. The Vulkan shader admission rules were not widened to bypass an unsupported unpack instruction.

[Native Whisper performance contract](../speech/whisper-go-performance-contract-20260930.md).
