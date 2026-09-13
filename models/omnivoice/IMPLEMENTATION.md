# Native inference milestone: backbone, denoising, decode

## Implemented and tested

- Full mixed text/audio forward: each audio position sums the eight codebook
  embeddings, text uses codebook-zero text IDs, then all 28 Qwen3 layers, final
  RMSNorm and chunked audio-head projection. Returns `[codebook,time,vocab]`.
- Fixed-capacity `Backbone.ForwardInto` streams weights using the reusable arena;
  all reads and calculations are native Go. Zero successful-call allocations.
- Sampler: float32 shifted timetable, reveal counts, normalized classifier-free
  guidance, mask-token suppression, top-k, Gumbel perturbation, confidence/layer
  penalty and filled-position masking. Scratch is preallocated.
- `Generation.GenerateInto` connects conditional/unconditional forwards to the
  reveal loop. Go PCG RNG is seeded per invocation, not PyTorch-compatible.
  Position temperature is constant as in upstream, not annealed.
- Decode-only HiggsAudioV2: RVQ embedding/projection sum, fc2, DAC input conv,
  five transposed-convolution upsampling blocks, residual convolutions with
  dilations 1/3/9, Snake activations and output convolution. Final tanh is omitted
  as required by Higgs. SIMD GEMM handles convolutions using bounded im2col tiles;
  scatter and sine remain scalar. Current decoder allocates per operator.
- Validated fixed codec variant: 24 kHz, 8 quantizers, codebook size 1024,
  decoder rates 8/5/4/2/3. Decode limit is 250 frames (10 seconds).
- WAV output exclusively creates PCM16 mono files and attenuates only to avoid
  clipping. Original waveform generation is distinct from file gain scaling.

## Numerical evidence

| Check | Result |
|---|---|
| Tiny full forward versus upstream OmniVoice.forward | every logit, max error 2.39e-7 |
| Real full forward, 3 mixed positions | all 24,600 logits, max error 0.000153, mean 1.34e-5 |
| Real forward allocation counters | 0 bytes, 0 allocations after setup |
| Tiny 4-step greedy generation | matches upstream-derived loop output/schedule |
| Sampler random primitives | upstream helpers with explicit uniform fixtures |
| Tiny complete generation allocations | zero after construction |
| Real decoder, 8 codebooks × 2 frames | all 1,920 samples, max error 5.59e-7 |

The real forward took ~2.34 seconds with streamed weights. The Python figure
includes model/codec loading, so these timings are **not** a valid speedup ratio.
Sampling tie order can differ from Torch: Go uses lowest flattened index.

The top-k ratio uses float64, matching Python scalar semantics. In float32,
`0.1 * 20` followed by double-precision `ceil` can produce 3 instead of 2; a
reference fixture exposed this and the API was corrected before integration.

## CLI

```sh
make test-omnivoice vet-omnivoice build-omnivoice
bin/omnivoice -mode logits -model "$MODEL" -input token-input.json
bin/omnivoice -mode generate -model "$MODEL" -input prepared-prompt.json \
  -steps 8 -output synthetic.wav
```

`logits` input: `tokens`, flattened `[codebook,time]` `ids`, `audio_mask`, optional
`positions` and additive `mask`. CLI limits to 256 positions.

`generate` input: `conditional` and `unconditional` objects with `tokens`, `ids`
and `audio_mask`; plus `target_frames`, descriptive `text` and `reference`.
Target frames occupy the end of each sequence. Maximum input length is 512
positions. It speaks the **supplied tokens**; descriptive text is not verified
against them and is not tokenized by this command. Generate currently uses full
attention/implicit positions, so custom masks/positions are not supported.

## Preparation boundary

`scripts/omnivoice-export-prompt.py` uses the existing Python reference runtime
to tokenize one text and encode one reference WAV. It writes a prepared JSON.
This is a development bridge, not part of the Go executable and not a claim of
fully native arbitrary text/reference input. Go runs all subsequent iterative
inference, codec decoding and WAV writing without Python subprocesses.

Keep reference-derived prepared prompts private just like reference recordings.
They are not fixtures/source exports. The checked-in tiny checkpoint is random
synthetic test data, and the decoder fixture uses synthetic code IDs.

## Real native synthesis and packed SIMD

Two 3-second synthetic Nimoy-conditioned samples completed using one prepared
prompt, eight steps and seed 42. Both external ASR checks recovered “The evidence
is insufficient, Captain.” Native baseline took 496.999 s (492.631 s generation,
4.368 s decode/save). Packed SIMD took 149.283 s (143.607 s generation, 5.675 s
decode/save), about 3.3× faster in these runs. This is not a controlled comparison
with Python, whose RNG and postprocessing differ. Peak observed generation RSS
was roughly 1.06 GB, not a formal whole-process maximum measurement.

Added `SgemmNTPackedTo`: checked slices, caller-owned panel scratch, existing
amd64 6×16/arm64 4×16 GEBP microkernel and safe tail fallbacks. This avoids repeated
B dot-product scans across rows. Shape/alpha/zero-allocation tests and race tests
pass. Real 128-token block fell from ~766 ms to ~126 ms with zero allocations;
real-block prefix/aggregate parity remained within float32 tolerances. Tiny full
logit fixtures use a small-row fallback; broader real packed-path parity is
covered by the 128-token block check, not the three-token logits check alone.

Full `go test ./backends/...` was run because a reusable backend entry point was
added. Existing SpacemiT package build/test failures reproduce on untouched
`e4c24e6f`. SIMD runtime tests pass; no architecture assembly was changed.

## Still required for completion

1. Native tokenizer/prompt contract validated against Qwen's exact regex and
   special tokens. Existing generic tokenizer is close but its digit grouping
   needs review for this model; do not claim arbitrary-text equivalence yet.
2. Native reference encoder (DAC acoustic encoder, HuBERT semantic feature path,
   semantic encoder, RVQ quantization) or a clearly scoped cached-voice import
   contract approved by the user. The current codec is decode-only.
3. Decoder scratch reuse, profiling of actual generation, matrix layout/packing
   optimization and SIMD exponential/sine/conversion assessment.
4. Text/reference → native speech listening acceptance against approved sample 3.
5. Broader shape/error tests, long utterances/chunking, optional post-processing
   matching upstream (silence removal, reference RMS gain, fades/padding).

The goal remains active. This is a working staged implementation, not a completed
fully native replacement or a full-SIMD graph.
