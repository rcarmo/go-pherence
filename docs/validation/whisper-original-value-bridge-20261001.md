# Original Whisper model-value bridge — 1 October 2026

Native Go loads the retained original Q5_0 Whisper model and matches its JFK, Portuguese and French content tokens/text/segment times on bounded fixtures. Weights widen to F32, so this diagnostic bridge provides numerical compatibility groundwork and no packed-Q5 speed improvement. The original performance goal is incomplete.

## Storage and mapping

The [legacy GGML loader](../../loader/whisperggml/README.md) reads the pinned original file directly. The supported retained models are:

- Q5_0: `394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2`, header file type 2008.
- Mixed F16/F32: `1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69`, header file type 1.

Both have 587 tensors, 32 encoder/four decoder layers, 1,280 hidden width, 20 heads, 128 mel bands, 1,500 encoder positions, 448 decoder positions and vocabulary size 51,866. Header/filter/vocabulary/tensor geometry is checked before model loading. Q5_0 blocks are decoded using the existing native Go GGML block decoder, with original dtype and offsets retained in loader metadata. No inference wrapper or requantisation is used.

The private [source bridge](../../model/whisper/legacy_ggml_source.go) maps names to HF conventions and reverses dimension metadata; row-major payload order remains unchanged. Original convolution bias `[out,1]` becomes HF rank 1. The checked loader receives logical F32 metadata describing decoded values, while original storage type/hash remain explicit in the evidence. Header/config mismatch, unexpected/duplicate mapped names, footprint overflow and checked tensor inventory errors fail before successful model publication.

Pinned HF config, greedy generation policy and tokenizer are used explicitly. All 50,257 original content vocabulary entries agree byte-for-byte with single-token HF decoding. Special-token policy comes from the pinned HF generation metadata; whole original decoder policy equivalence is unqualified. The Go frontend remains its checked 400-point Slaney implementation rather than switching to embedded model filters.

## Independent and trained gates

A small independent C++ oracle calls `ggml_get_type_traits(GGML_TYPE_Q5_0)->to_float` from retained whisper.cpp revision `c44b60b8053bbf2a5c1e014f11323fb3f2485177`. Sixty-four blocks/2,048 values cover signed/zero/subnormal/large finite scales and varied low/high quantisation bits. Native Go agrees bit-for-bit for all values in ten repetitions. Header, library, oracle binary, input and output hashes are retained in the [manifest](../../benchmarks/speech-foundations/whisper-original-value-bridge-20261001/manifest.json).

The Q5_0 and F16 models both load successfully in guarded isolation. Q5_0 trained Go Vulkan key32/tile64 runs use five repetitions per fixture. All repeats match their first result. Against the original Q5 output, content token IDs, trimmed text and segment start/end times agree:

| Fixture | Go request median | Original segment bounds |
|---|---:|---:|
| JFK | 7.442 s | 0–10.400 s |
| Portuguese 0 | 7.421 s | 0–7.360 s |
| French 0 | 6.886 s | 0–2.420 s |

Original timestamp boundary tokens are compared separately from content tokens. Original Portuguese/French quality arms ran once; these are output compatibility checks, not repeated timing acceptance. Original JFK repeated engine evidence is retained in the earlier [tile64 campaign](vulkan-linear-regtile64-20261001.md). Go request time excludes loading/preparation and still uses F32 Vulkan storage/accumulation; original Q5 uses packed quantisation and different numerical/storage boundaries. Q5/F16 load time is about 5.2/6.7 seconds in this widened source bridge.

Parser tests run ten times and cover malformed bounds/types/names, duplicate/truncated records, pin mismatch, metadata ownership, nonfinite payloads, closure and cancellation on every metadata read. A final-read cancellation bug found by this test was corrected: no partial parsed object escapes. Full-tree build/vet/tests/race, layout, ARM64/RISC-V cross-builds and documentation/shader tests pass.

## Rejected kernel experiments

Native alternating six-sample-per-kernel tests retain bit-identical outputs and guard checks, but qualify no additional whole-request gain:

- Packed F16 weights with hardware-assisted pair unpack remain 1.3–5.0% slower than current F32 tile64 on the three turbo projection shapes. Manual unpack is slower still. A temporary narrowly typed `UnpackHalf2x16` shader admission experiment was reverted; no admission expansion is committed.
- Q5_0 generic unpack is 31–38% slower. Block-wise shared dequantisation is mixed/near-neutral: about +3.8%, −0.7%, −2.6% kernel time for the three shapes. It has no trained packed inference qualification and was not adopted.
- Shared-memory padding, K8/K16/K64, 128-column output, smaller rectangular output tiles, altered lane geometry, full unrolling and input/weight transposes were neutral or slower. Reduced-query attention tiles were slower. Explicit exact FMA projection was neutral.
- Vector-output shader experiments were rejected by the existing opcode admission envelope. Two generated diagnostic variants initially failed native correctness through incorrect tail-row/output-channel addressing; they were corrected and rerun, then rejected on performance. Failing outputs were never used for acceptance.

All external-shader test sources were removed and production/defaults are unchanged. This work establishes an explicit path to the original quantised values for future native packed kernels. Independent acoustic/word-timing accuracy, natural overlap, long recordings and matched original speed remain unqualified.

[Native Whisper performance contract](../speech/whisper-go-performance-contract-20260930.md).
