# OmniVoice native CPU port

Native text/reference-WAV→speech execution includes HuBERT/DAC reference encoding, Qwen3 denoising and HiggsAudioV2 decoding. The Go executable does not invoke Python. References require an explicit transcript. `-preprocess-reference` enables upstream-compatible reference silence trimming; `-postprocess` trims generated silence and applies fades/padding. Both default off for reproducible raw output. `plan-chunks` previews bounded text partitions; `synthesize-long` generates and joins them using a fixed reference, 5 ms edge fades and 100 ms gaps. This boundary policy is native and differs from upstream chunking. See [IMPLEMENTATION.md](IMPLEMENTATION.md) for parity, profiling and remaining work.

`NewBackboneSibling` shares a streamed weight arena for sequential classifier-free-guidance branches. Never run sibling forwards concurrently. Backbone, codec and reference workspaces are single-caller; output/input alias restrictions in each API are caller obligations. Long synthesis reserves the largest planned backbone, generation and decoder shapes once, then reconfigures active views per chunk. Reconfigure both backbones before their generation workspace. Post-processing and retained output copies still allocate.

## Implemented

- Typed OmniVoice/Qwen3 config and header-only safetensors checkpoint validation in `loader/omnivoice`.
- F16/F32/BF16 floating weight acceptance, I64 codebook offsets, expected names and dimensions. Single-file and sharded lazy weight access; only one requested layer is converted to float32 for the probe. Inspection and execution both prefer the shard index when present.
- Batch-one Qwen3 decoder block: input RMSNorm, Q/K head RMSNorm, default split-half RoPE, non-causal grouped-query attention with optional additive mask, output projection, residuals, SiLU-gated MLP.
- SIMD GEMM, normalization and SiLU dispatch, exact F16C conversion and bounded-error AVX2/FMA exponential for attention softmax. Go orchestrates token/head packing and rotary positions; SiLU uses SIMD exponential and separate vector add/divide/multiply on amd64; Snake, encoder ELU and HuBERT erf-GELU use bounded SIMD kernels with scalar fallback.
- `go-264/audio` frontend, pinned at `48ff0ca8272a`, for bounded reference decoding/channel conversion/resampling to 24 kHz mono.
- CLI modes include `plan-chunks`, `synthesize-long`, `prepare`, `encode-reference` and `synthesize` (raw or cached reference), plus `inspect`, `block`, `stack`, `logits`, `generate` (prepared prompt), `audio` and `capabilities`.
- Allocation-free `ForwardInto` and reusable layer-weight arena; full 28-layer synthetic-input profiling with CPU/allocation profiles. See [PROFILING.md](PROFILING.md) for measurements, limits and commands.
- Explicit `-backend auto|cpu|vulkan` policy; hardware/software Vulkan detection, CPU fallback in auto and rejection of unimplemented explicit Vulkan execution.

## Build and test

```sh
make test-omnivoice vet-omnivoice build-omnivoice
bin/omnivoice -mode inspect -model /path/to/OmniVoice
bin/omnivoice -mode block -model /path/to/OmniVoice -layer 0 -tokens 3
bin/omnivoice -mode audio -reference /path/to/reference.wav
```

The block probe uses synthetic hidden states, not text or speech tokens. Default is two Go execution threads; GEMM kernels are invoked serially. It must not be treated as a TTS latency benchmark.

`testdata/omnivoice/tiny-block.json` is a deterministic PyTorch/Transformers fixture. Go tests compare every output element for full attention and for an explicit blocked-key mask. Maximum observed absolute errors on amd64: 5.96e-8 and 8.94e-8. Another test confirms that a future token affects earlier output (non-causal behavior).

Optional real-checkpoint checks:

```sh
GO_PHERENCE_REAL_OMNIVOICE=/path/to/OmniVoice go test ./loader/omnivoice -run TestRealCheckpointMetadata
/path/to/python scripts/omnivoice-real-block-parity.py --model /path/to/OmniVoice
```

The real-block script loads one layer only. It checks the first eight outputs, total sum and peak against PyTorch; it is not full-tensor parity. Measured with the locally stored float16 OmniVoice checkpoint, layer 0 and three synthetic tokens: first-eight error 4.16e-7, peak error 4.96e-7, sum error 2.09e-5. Native forward took about 53 ms, excluding roughly 141 ms loading. These numbers do not predict full utterance speed.

The approved Charlie X reference decodes through go-264 to 108,000 samples (4.5 seconds). Native reference encoding with RMS normalisation and silence preprocessing reproduces all 832 codes (104 frames) from the Python reference cache.

## Remaining work, in dependency order

1. Remaining scalar kernels: out-of-range/fallback Snake sine, exceptional/range-fallback sampler exponentials, GELU/ELU/Snake fallback regions and non-amd64 SiLU finishing. Hardware Vulkan execution is not implemented.
2. Stochastic whole-loop equivalence under supplied noise and broader multilingual speech validation. Token IDs match the upstream multilingual fixtures, including NFC decomposition; pronunciation has not been evaluated across those languages.
3. Listening acceptance of native samples and chunk transitions. Keep assistant integration out of scope until that gate passes.

Unsupported config variants fail at block construction (biases, non-SiLU, non-default RoPE, sliding windows, incompatible GQA). No CUDA, CGo, subprocess inference or model download is required by the Go CLI. Measured short native synthesis takes roughly 83–94 seconds for about three seconds of audio on this VM, depending on preprocessing and build. These runs do not establish a speedup over PyTorch.

Affected-package tests, race tests, vet and the CLI build pass, including `CGO_ENABLED=0`. A full `go build ./...` fails in existing SpacemiT host stubs/C compiler flags and DiffusionGemma command APIs. The same errors were reproduced in an untouched worktree at upstream `d08ce322`; no unrelated repairs were included.

### Shared CFG traversal experiment

Add `-shared-traversal` to streamed `generate`, `synthesize`, `synthesize-long`
or `serve` runs to load each decoder layer once for both guidance branches.
This keeps separate full attention and exact SIMD row shapes. It requires the
streamed float32 path: do not combine it with `-resident-mib`, `-prepack` or
`-direct-q8`. The default remains separate traversal. Two local timing pairs
showed mixed results; see [the profiling measurements](PROFILING.md).

Library callers select `GenerationConfig.SharedTraversal` with distinct
`NewBackboneSibling` backbones. The sibling single-owner/sequential execution
contract still applies; neither backbone may execute concurrently elsewhere.

### Guidance quality experiment

`-guidance` selects the nonnegative CFG scale for `generate`, `synthesize`,
`synthesize-long` and `serve`. The default is still **2**. Scale 0 skips
unconditional inference; any positive scale runs both branches. Consequently,
lowering 2 to 1 changes audio but does not reduce transformer work.
`-guidance 0` cannot be combined with `-shared-traversal`.

The worker setting is fixed for its process lifetime and appears in its ready
record. Single/long synthesis JSON also reports the effective scale. No
per-request override is provided, so a worker's phrase cache cannot mix scales.

A single short English comparison measured 37.13 s at scale 0 versus 60.26 s
at scale 2. Both transcribed exactly; the scale-0 waveform RMS was less than
half the baseline. This experiment has not passed voice/prosody, long-text or
Portuguese listening gates. See [PROFILING.md](PROFILING.md) for the measurements
and [TRAINING.md](TRAINING.md) for the reuse/distillation assessment.

### Column-parallel worker experiment

Use `-gemm-workers 2 -gemm-columns` to partition weight panels among workers and
avoid duplicate streamed packing. The default remains row-parallel workers.
This works with streamed, resident, prepacked and shared-CFG execution; the
worker count and column selection are reported in synthesis/worker metadata.
Two local streamed timing pairs showed an 8.9% lower mean with byte-identical
audio; see [PROFILING.md](PROFILING.md). Library callers can use
`Backbone.EnableColumnWorkers` before creating siblings; existing siblings keep
their previous selection.

Measured combined-mode candidates on the two-vCPU N100 VM:

| Preference | Explicit flags | Three-second sample mean | Peak RAM |
| --- | --- | ---: | ---: |
| Lower memory | `-gemm-workers 2 -gemm-columns -shared-traversal` | 57.47 s | 1.06 GiB |
| Lower latency, more RAM | `-gemm-workers 2 -gemm-columns -resident-mib 2048` | 51.58 s | 2.66 GiB |

These are two-trial single-shot measurements with eight steps and guidance 2;
all ten mode-comparison WAVs match. Timing noise includes a retained columns-only
outlier. Prepacking did not win total command time. See the full
[measurements](PROFILING.md#combined-workercache-measurements-2026-09-14-1ec0bc70)
before choosing settings for a persistent worker or longer utterances.
