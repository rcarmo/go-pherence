# b607 direct benchmark audit

Frozen source revision: `065d9d50152486590c09b31627ecaf76ceba39dd`.

## Finding

The quarantined Go experiment correctly reproduces one b607-style 8-row by 4-token tile, but its supposed 16-token projection is four serial calls to that tile. It therefore does not benchmark b607's fused AVX2 8-row by 16-token supertile and cannot reject that topology.

## Call topology

Frozen b607 `gemm_q4_b32_8x8_q8_0_lut_avx`:

- forms four `block_q8_0x4` pointers;
- allocates 16 YMM accumulators;
- loads and decodes each 144-byte `block_q4_0x8` panel once per QK block;
- consumes all four Q8 panels inside the decoded-Q4 lifetime; and
- writes 16 token rows by eight weight rows.

The Go experiment's `go_llama_q4_0_q8_0_projection` instead loops over four token groups and calls `go_llama_q4_0_q8_0_8x4` once for each group. Each call reloads the 144-byte Q4 panel, rebuilds all Q4 permutations and nibble decodes, reloads the eight FP16 weight scales, and starts four accumulators from zero. Grouping those calls under a `super` loop does not fuse them.

## Equal-work accounting

For one 8-row by 16-token by 32-K block:

| Work | Frozen fused b607 | Quarantined Go projection |
|---|---:|---:|
| Output dot products | 128 | 128 |
| Q8 panels consumed | 4 | 4 |
| Q4 panel loads | 1 | 4 |
| Q4 panel permutations/decodes | 1 set | 4 sets |
| Weight-scale loads/conversions | 1 set | 4 sets |
| Persistent output accumulators | 16 | 4, restarted four times |

The integer dot count and Q8 payload are not reduced by fusion. The mechanism is removal of three repeated Q4 loads, decodes and scale preparations while 16 output accumulators remain live.

## What the retained measurements actually establish

The 2,976 ns tile result compares one 8-row/4-token b607-style call with four retained 1-row/8-token calls -- 32 outputs on each side. It says that the four-token subkernel is slower for that equal-work shape.

The 1.525595 ms prepacked and packing-inclusive projection results repeatedly invoke that same four-token subkernel. They do not measure the frozen fused loop. These numbers remain valid rejection evidence for the serial orchestration, but the previous conclusion that they reject the fused b607 topology is invalid.

The layout, deterministic topology-reference and intentional non-exactness tests remain useful and correct. Production also correctly remained on `dotQ4_0Q8_0Tokens8SoAVNNI`; no second 2.067 GiB weight representation was created.

## Corrected gate

Before model integration, a valid candidate must:

1. implement one 8-row/16-token AVX2/AVX-VNNI call with 16 live accumulators and one Q4 decode per block;
2. match a portable fused-topology reference for all FP32 outputs and padded tails;
3. compare 128 equal outputs against the retained kernel and run the existing 128-row/124-token projection benchmark;
4. include Q8 packing in the projection result; and
5. show a mechanism-backed, clear gain before any replacement-only model packing is considered.

Four serial 4-token calls are not an acceptable substitute for this gate.
