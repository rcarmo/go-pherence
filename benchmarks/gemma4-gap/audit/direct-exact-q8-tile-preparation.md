# Direct exact Q8 tile preparation audit

## Status

Accepted as part of the promoted exact row-block plus AVX2 preparation batch. It was first correctness-gated in isolated worktree `/tmp/go-pherence-direct-q8` at baseline `1c1ff55d`, then layered only after traversal timing was independently accepted. The original reproducible source diff is [`direct-exact-q8-tile-preparation.patch`](direct-exact-q8-tile-preparation.patch).

## Current retained preparation

For `batch=124`, `inDim=2560`, there are 80 Q8 blocks per token:

1. `quantizeQ8_0BatchTo` writes 9,920 token-major `q8_0Block` values;
2. the long-prefill branch copies the first 120 tokens into 1,200 `q8_0Tile8` values;
3. four token-major rows remain for the four-token tail.

The structures are 36 and 288 bytes respectively. Current transient activation storage is therefore:

| storage | calculation | bytes |
|---|---:|---:|
| token-major Q8 | 124 x 80 x 36 | 357,120 |
| eight-token SoA tiles | 15 x 80 x 288 | 345,600 |
| total | | 702,720 |

For the first 120 tokens, preparation writes 345,600 token-major bytes and then reads and rewrites the same 345,600 bytes into SoA form.

## Proposed mechanism

Quantise each 32-value F32 block with the unchanged `quantizeQ8_0To` block operation, but write its FP32 scale and 32 signed bytes directly to the final `q8_0Tile8` token slot. Quantise only the four-token tail to ordinary `q8_0Block` storage.

The final storage becomes 345,600 tiled bytes plus 11,520 tail bytes, or 357,120 bytes. This removes 345,600 allocated bytes and one 691,200-byte write/read traversal per projection. It does not remove the min/max, rounding, clipping or FP16-roundtrip quantisation work.

This is distinct from `quantizeQ8_0x4To` in the fused 8x16 experiment. That helper writes the frozen llama-style non-exact activation panel consumed by the ceiling kernel. The proposed helper writes the retained exact `q8_0Tile8` bytes and feeds the unchanged exact assembly.

## Exactness argument and gates

Byte identity is achievable because `q8_0Tile8` is only a destination permutation:

```text
tile.d[token]   = tokenMajor[block].d
tile.qs[token]  = tokenMajor[block].qs
```

No arithmetic, reduction or scale representation changes. Before timing, require:

- direct output byte-for-byte equal to quantise-then-pack for random input;
- zero-scale, clipping and FP16-roundtrip edge cases;
- block counts 1--80;
- real `batch=124`, `inDim=2560`, including the four-token tail;
- focused repetitions, race, full `loader/gguf` package, vet and non-amd64 compile gates;
- exact projection comparison at 513 rows and 124 tokens.

## Expected value

The mechanism halves transient activation storage and removes a redundant 691,200-byte traversal, but leaves quantisation arithmetic and the dominant projection kernel intact. Its expected end-to-end effect is therefore incremental rather than gate-closing. It should be layered only after the cache-blocked traversal is independently measured, and retained only if the combined packing-inclusive projection and frozen request improve cleanly.

No historical audit found an exact direct-to-`q8_0Tile8` implementation or timing. Existing direct-preparation evidence belongs to the non-exact fused layout and cannot accept or reject this exact destination permutation.

## Correctness result

The isolated implementation factors the unchanged scalar block operation into `quantizeQ8_0BlockTo`, writes full groups directly to `q8_0Tile8`, quantises the sub-eight tail to ordinary blocks, and uses atomic group claims with five goroutines plus the caller at six-way execution.

Byte equality passed for `(batch,width)` values `(64,32)`, `(65,256)` and the real `(124,2560)` shape, including zero blocks, random clipping-scale values and the four-token tail. Projection-oracle checks also pass. Saved output is [`direct-exact-q8-tile-preparation-correctness.log`](direct-exact-q8-tile-preparation-correctness.log):

```text
GOMAXPROCS=6 go test ./loader/gguf -count=1                         ok
focused direct preparation and projection oracle -count=20         ok
focused race -count=3                                               ok
go vet ./loader/gguf                                                ok
CGO_ENABLED=0 GOOS=linux GOARCH={arm64,riscv64} go test -c          ok
```

## Timing result

In the clean three-way packing-inclusive comparison after accepted row blocking, direct-to-SoA alone was effectively flat: retained and direct-scalar medians were 10.107 and 10.141 ms/op. It nevertheless reduced transient allocation from about 714 KiB and 26 allocations/op to 365 KiB and 19--20 allocations/op. Direct preparation was retained only in combination with the exact AVX2 quantiser, whose longer paired comparison established a clear projection gain. Raw three-way evidence is [`go_q4_row_block_direct_q8_three_way.log`](go_q4_row_block_direct_q8_three_way.log); its load monitor identified one Jupyter process in the first scalar sample, which does not change the flat conclusion.
