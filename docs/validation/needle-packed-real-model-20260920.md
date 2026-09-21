## Needle 3 packed projections and released-model smoke tests

The direct CQ path now runs matrix products from packed indices and FP16 norms, without reconstructing the full matrix on every multiply. It transforms each padded input group with normalized Walsh-Hadamard, unpacks codebook values into a 128-float working block, and uses the SIMD dot kernel. Binary, 2/3/4-bit and ternary group-128 layouts have independent dense-oracle tests.

These measurements precede the subsequent [allocation/pprof optimisation pass](needle-allocation-pprof-audit-20260920.md); they are preserved as its baseline.

This is a hybrid execution path. Attention projections, engram projections and the tied output head use original packed archive matrices; other operations and checkpoint export retain decoded tensors. Consequently `-packed` is opt-in and does not reduce total model memory yet. mHC and auxiliary-head projection work still uses decoded tensors. `PackedBytes()` reports the additional retained CQ payload/codebook bytes, not total RSS.

## Checked paths

The packed matrix constructor owns its input, validates shapes, bit widths, codebooks, norms and ternary codes, and bounds geometry before copying. Multiplication rejects invalid lengths, input/output aliasing and nonfinite inputs before output writes. Padding and batch tails are covered. Scratch for the input transform is included in the model workspace charge.

Archive parsing retains owned original CQ payloads alongside decoded data and counts both against its limit. Fresh file loads transfer private decoded buffers into model ownership, avoiding a redundant full copy; the public checkpoint constructor and caller-provided archive mapping continue to copy. The conservative load plan remains bounded at 1 GiB of logical buffer storage. That is not an RSS ceiling: allocator retention, runtime overhead and the decoder are separate.

Packed and decoded logits agree within the existing numerical bounds for full-prefix, cached and depth-sliced model execution. Greedy fixture output is identical. Tests include scalar-disabled dispatch, source-checkpoint rejection, decoded-checkpoint rejection, caller-data ownership, cache rollback and the CLI's packed flag. An independent delegate review timed out and supplied no findings; it adds no review coverage.

## Released archive

Model: `Cactus-Compute/needle3`, Hugging Face revision `b274efcb211a9eef48c9a88da4b43bd569696a39`, Apache 2.0. Archive SHA-256:

```text
c9d915eca282ed42d1a09b143b592adb4cc6744ffe2d294adf5cfc5548170c38
```

The 35,335,380-byte archive has 20 layers, width 768, 12 attention heads, two KV heads, QK width 48 and V width 64. Its KV window is 256 tokens. It contains 115 two-bit CQ records and seven four-bit records, plus FP16/FP32 and tokenizer records; the decoded tensor inventory is 484,087,640 bytes. The downloaded archive/config live under ignored `checkpoints/needle3/`.

A raw `Hello` prompt with archive BOS, cached decoding and two generated tokens produced `[8110,417]` (`, I`) on both Intel decoded and packed paths. Cold process timings with `GOMAXPROCS=2`, including model loading:

| Path | Wall time | Peak RSS |
|---|---:|---:|
| Decoded | 1.23 s | 914,944 KiB |
| Packed hybrid | 1.82 s | 916,480 KiB |

A second prompt used the exact text below, with archive BOS prepended by the CLI:

```text
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
```

Twelve tokens with EOS stopping disabled were identical on all four paths: Intel decoded, Intel packed, native CIX P1 ARM64 decoded and native ARM64 packed:

```text
[6,38,8141,1515,4047,997,598,782,326,1118,296,1957]
<think>
History shows user was about to start a video
```

The odd short continuation is recorded as-is. It proves neither task quality nor agreement with the native upstream C++ engine. It establishes that the released checkpoint loads, text tokenization runs, and the four Go execution paths agree on this bounded case. Intel cold process timings were 1.62 s decoded and 2.76 s packed; peak RSS was 1,093,116 and 1,093,632 KiB respectively. No ARM speed comparison was taken on the shared board.

## Performance and limits

A synthetic 576-by-768 four-bit matrix/vector multiply took about 0.534 ms and 3,072 bytes/one allocation on the Intel i7-12700. On the tiny archive's twelve-token cached sequence, decoded execution took 1.252 ms versus 2.552 ms packed. Packed is slower here, so it is not the default. These short measurements include no warmed real-model throughput study.

Native ARM fixture runs passed for model, loader, CLI and SIMD packages. Direct packed products still use scalar bit unpacking and Walsh butterflies around SIMD dot products; full hot-path vectorisation, packed-only loading, global KV eviction, schema-constrained tool generation and wider real-model validation remain open. None of these runs used a GPU, changed services or resumed the frozen evaluation.
