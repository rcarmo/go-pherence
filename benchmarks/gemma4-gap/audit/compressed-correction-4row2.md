# Exact compressed-correction four-row/two-token candidate

## Status

**Rejected by audit before implementation.** Historical exact output-major measurements provide a stronger bound than the byte-count hypothesis alone.

## Mechanism

The retained long-prefill kernel owns eight YMM FP32 accumulators for one weight row by eight tokens. That minimises Q4 decode instructions but reloads every Q8 vector and dynamically recomputes every correction for each output row.

The existing exact four-row/two-token kernel has the complementary reuse direction, but its older signed-byte formulation repeats `VPABSB` and `VPSIGNB` for every output. A compressed-correction rewrite would:

1. pack two Q8 scales, two 32-byte Q8 vectors and two exact int16 correction vectors per block;
2. load/sign-extend each correction once with `VPMOVSXWD`;
3. retain both Q8 vectors and both corrections while decoding each of four Q4 rows once;
4. issue eight unsigned-Q4 `VPDPBUSD` instructions, subtracting the retained correction for the matching token;
5. preserve one YMM FP32 accumulator per logical output and the legacy final reduction order.

The exact correction bound is [-4096, 4064], already proved by the one-row/eight-token candidate.

## Register budget

AVX2's 16-vector-register limit is met without spills:

| registers | use |
|---|---|
| `Y0`--`Y7` | eight persistent output accumulators |
| `Y8` | decoded unsigned Q4 row, reused by both tokens |
| `Y9`--`Y10` | two Q8 vectors, reused by all four rows |
| `Y11` | integer dot / FP32 dot temporary |
| `Y12`--`Y13` | two sign-extended corrections, reused by all four rows |
| `Y14` | nibble mask |
| `Y15` | FP16/FP32 scale temporary |

This allocation is only possible with precomputed correction: dynamic correction needs a persistent `0x08080808` vector and exceeds the register file if both corrections are retained.

## Data movement per QK block and eight outputs

| orientation | Q4 payload/scale | Q8 tile | total source bytes | VNNI dots |
|---|---:|---:|---:|---:|
| retained one-row/eight-token dynamic | 18 | 288 | 306 | 16 |
| one-row/eight-token int16 candidate | 18 | 416 | 434 | 8 |
| proposed four-row/two-token int16 | 72 | 104 | 176 | 8 |

The proposed orientation cuts source traffic by 42.5% versus retained dynamic correction and by 59.4% versus the pending one-row/eight-token int16 candidate for the same eight outputs. It does decode four Q4 rows rather than one, so instruction count and complete packing-inclusive timing remain decisive.

For the real 124-token, 80-block shape, 62 token-pair tiles require 515,840 bytes. This remains transient and introduces no duplicate model weights.

## Historical bound and decision

Commit `57301e7a` already measured the exact output-major four-row/two-token orientation on this host. It passed 100 deterministic cases over blocks 1--80 and all 64 FP32 lane states, but its projection median was 3.151859 ms versus 1.876609 ms retained--68.0% slower. Its single-tile median was 1,545 ns versus 615.5 ns.

The existing output-major function disassembles to 228 instructions with eight `VPDPBUSD`, eight `VPABSB` and eight `VPSIGNB`. The proposed compressed correction removes the 16 absolute/sign instructions but adds two `VPMOVSXWD`; unsigned-Q4 correction subtraction replaces the eight existing nibble-bias subtractions rather than removing them. The optimistic static total is therefore about 214 instructions--only 6.1% lower--while payload rises from the historically tested 144 bytes to 176 bytes per eight outputs.

Even granting an unrealistically proportional 6.1% speed-up gives approximately 2.96 ms, still 57.7% slower than retained. Beating retained would require a 40.5% reduction from the measured output-major time. The register fit is valid, but the mechanism is far too small to overturn the measured orientation penalty, so implementation would violate the ranked-hypothesis rule.

## Correctness gates (not run)

- Random blocks 1--80 must match eight scalar outputs bit-for-bit.
- Capture and compare all eight lane accumulators before reduction where practical.
- Preserve token-major output placement and row-tail fallback.
- Run focused tests repeatedly, race testing, vet and non-amd64 cross-builds.
- No model timing before the projection gate passes.

## Performance gate (not run)

Switch the production long-prefill scheduler coherently to four-row groups and two-token tiles, including tile packing and output scatter. Compare five alternating pinned samples against detached retained baseline `1c1ff55d`:

```sh
taskset -c 0-5 env GOMAXPROCS=6 go test ./loader/gguf \
  -run '^$' \
  -bench '^BenchmarkQuantMatrixProjectBatchQ4_0Gemma4Shapes/out10240/batch124/batched$' \
  -benchmem -benchtime=1s -count=1
```

Require a clear packing-inclusive gain in uncontaminated pairs. Any exact candidate that produces a verified end-to-end improvement will be retained as incremental gap closure even if it does not reach the final 89.4050 prompt tok/s gate. It must still pass the exact 124+48 state/ID/logit/K/V gate before frozen end-to-end timing, documentation, commit and push.
