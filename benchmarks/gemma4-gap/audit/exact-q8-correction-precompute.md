# Exact Q8 correction precomputation audit

## Purpose

The fused b607 8x16 topology remains a performance ceiling, not a production candidate: its blockwise scalar accumulation cannot preserve the retained kernel's eight independent FP32 lane accumulators and final reduction order.

This batch instead changes the retained exact eight-token AVX-VNNI path. For every transient Q8 block and token, preparation stores the eight int32 lane corrections

```
8 * (q8[4*l+0] + q8[4*l+1] + q8[4*l+2] + q8[4*l+3]), l = 0..7
```

The assembly then computes the unsigned-nibble dot once and subtracts the stored correction. The identity is unchanged:

```
sum((q4_unsigned - 8) * q8) = sum(q4_unsigned * q8) - 8 * sum(q8)
```

Corrections depend only on transient activations, so each is produced once and reused for every output row. No weight representation changes and no permanent duplicate model storage is introduced.

## Layout and cost

`q8_0Tile8` is an assembly-visible 544-byte structure:

- scales: offset 0, 32 bytes;
- eight Q8 vectors: offset 32, 256 bytes;
- eight correction vectors: offset 288, 256 bytes.

A compile-time-layout test asserts offsets 32/288 and size 544. At the real 124-token by 2560-wide shape, 120 full tokens require 1,200 tiles: 652,800 transient bytes versus 345,600 bytes before, an increase of 307,200 bytes per projection. The correction storage is released with the projection scratch and is not model state.

## Prior result and why this is a distinct retest

An earlier four-row/single-activation candidate stored correction lanes in each 36-byte Q8 block. Against the then-retained signed-byte kernel it expanded blocks to 68 bytes and regressed the isolated x4 median from 177 µs to 181 µs; commit `38088967` records its rejection. That result is a strong warning against assuming arithmetic removal is free.

The current comparison is narrower and uses a different retained kernel: one row by eight tokens, SoA activation tiles, dynamic unsigned correction via a second `VPDPBUSD`, and an 81.85% profile share. The retest asks whether replacing that second VNNI dot with a cache-resident correction load and software-pipelining independent token pairs helps this exact topology. It must still pass the complete packing-inclusive gate and will be rejected without a clear gain.

## Static kernel effect

GNU `objdump -d -Mintel` over otherwise identical test binaries reports for `dotQ4_0Q8_0Tokens8SoAVNNI`:

| variant | instructions | `vpdpbusd` | `vmovdqu` |
|---|---:|---:|---:|
| retained dynamic correction | 192 | 16 | 9 |
| precomputed correction | 187 | 8 | 17 |

The loop therefore trades eight L1 correction loads for eight dependent VNNI dot instructions per QK block while preserving the eight output accumulators. It also removes construction of the constant `0x08080808` vector. The intended mechanism is lower VNNI-port pressure; the extra transient data is expected to remain cache-resident while it is reused across rows.

The eight token bodies are software-pipelined in four pairs. Each pair loads two Q8 vectors and two corrections, issues independent dots into `Y11`/`Y12`, then converts and updates the original token accumulators in their original block order. The schedule uses the register freed by removing the correction constant plus existing temporaries; it adds no instructions or spills. GNU disassembly decodes all eight raw VEX instructions correctly and retains the 187-instruction function total.

## Correctness

Passed before timing:

```sh
GOMAXPROCS=6 go test ./loader/gguf -count=1
GOMAXPROCS=6 go test ./loader/gguf \
  -run '^(TestQ8_0Tile8AssemblyLayout|TestDotQ4_0Q8_0Tokens8SoARandomExact|TestQuantMatrixProjectBatchQ4_0.*)$' \
  -count=20
GOMAXPROCS=6 go test -race ./loader/gguf \
  -run '^(TestQ8_0Tile8AssemblyLayout|TestDotQ4_0Q8_0Tokens8SoARandomExact|TestQuantMatrixProjectBatchQ4_0.*)$' \
  -count=3
go vet ./loader/gguf
CGO_ENABLED=0 GOOS=linux GOARCH=arm64 go test -c ./loader/gguf -o /tmp/gguf-arm64.test
CGO_ENABLED=0 GOOS=linux GOARCH=riscv64 go test -c ./loader/gguf -o /tmp/gguf-riscv64.test
```

The random exact test covers block counts 1--80 and compares all eight outputs bit-for-bit with the retained exact reference. Both non-amd64 compile gates pass. Repository-wide unrelated failures are outside this gate.

## Performance gate

A detached worktree at `1c1ff55d` preserved the immediate retained baseline. After the six pinned CPUs remained 92--97% idle for 15 seconds, five alternating baseline/candidate samples exercised the real 124-token by 2560-in/10240-out production projection with:

```sh
taskset -c 0-5 env GOMAXPROCS=6 go test ./loader/gguf \
  -run '^$' \
  -bench '^BenchmarkQuantMatrixProjectBatchQ4_0Gemma4Shapes/out10240/batch124/batched$' \
  -benchtime=1s -count=5
```

The raw paired log is [`go_q4_exact_correction_projection_bench.log`](go_q4_exact_correction_projection_bench.log):

| variant | five samples (ms) | median | median bytes/op |
|---|---|---:|---:|
| retained dynamic correction | 12.915, 12.242, 19.499, 13.263, 13.302 | 13.263 ms | 714,951 |
| precomputed + two-token pipeline | 18.932, 18.879, 18.694, 16.027, 14.716 | 18.694 ms | 1,018,177 |

One baseline sample was visibly contaminated, but that cannot rescue the candidate: it lost the other four pairs by 10.6--54.2%, and its median is 40.9% slower. The measured 303,226-byte allocation increase agrees with the predicted 307,200-byte tile growth after allocator effects.

## Decision

**The int32 representation was rejected and removed.** Halving VNNI dots did not compensate for doubling the hot activation-tile traffic. The exact tests prove semantics, but the complete packing-inclusive gate decisively fails. No model run or production promotion is justified.

## Compressed int16 follow-up

The correction range is exactly bounded: four signed Q8 bytes sum to [-512, 508], so multiplying by eight yields [-4096, 4064]. A second candidate therefore stores eight int16 corrections per token and sign-extends them with `VPMOVSXWD` in the paired pipeline.

This reduces `q8_0Tile8` from the rejected 544 bytes to 416 bytes. At 1,200 real-shape tiles, added transient scratch falls from 307,200 to 153,600 bytes. GNU disassembly reports:

| variant | instructions | `vpdpbusd` | Q8 `vmovdqu` | `vpmovsxwd` |
|---|---:|---:|---:|---:|
| retained dynamic correction | 192 | 16 | 9 | 0 |
| compressed correction + pipeline | 179 | 8 | 9 | 8 |

Unlike the int32 attempt, the compressed path does not add 256-bit correction loads: each `VPMOVSXWD` consumes 16 bytes and produces the required eight int32 lanes. The output still matches the retained reference bit-for-bit for random block counts 1--80. Twenty focused repetitions, three race repetitions, the full package and focused vet pass. Its new packing-inclusive set is retained in [`go_q4_exact_correction_int16_projection_bench.log`](go_q4_exact_correction_int16_projection_bench.log), but is contaminated: alternating medians misleadingly favour the candidate by 4.0% while pair outcomes swing from a 67% loss to a 13% gain. The quietest baseline reached 12.053 ms while the best candidate reached only 15.608 ms.

More importantly, this representation had already received the stronger complete-request test recorded in [`go_q4_compact_correction_microbench.log`](go_q4_compact_correction_microbench.log) and [`go_prefill_compact_correction_rejected_paired_r3.log`](go_prefill_compact_correction_rejected_paired_r3.log). Its isolated kernel ran at 0.937--0.958 of dynamic-correction time, but the same 416-byte tile and packing lost every request pair: medians were 40.392 versus 44.863 prompt tok/s. Two-token software pipelining does not reduce that tile footprint or packing cost, and the current projection floor supplies no contrary clean evidence. The compressed candidate is therefore **rejected and removed** rather than repeating an already completed full-model experiment.
