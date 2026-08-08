# Gemma4 CPU performance gap

This programme compares `go-pherence` with a CUDA-disabled rcarmo/llama.cpp b607 (`065d9d501`) build on the same Gemma4 E4B QAT Q4_0 GGUF. The comparison is intentionally awkward: timings must describe the same phases, and Go's activations, logits, generated tokens, K/V state and FP32 lane reduction order must not move by a bit merely to make a chart look better.

The earlier 447.17 prompt tok/s result did exactly that last part wrong in a more mundane fashion--it came from a CUDA-enabled binary where `-ngl 0` did not produce a CPU-only execution. It is retained as [`llama-cuda-ngl0-invalid.json`](llama-cuda-ngl0-invalid.json), with an unambiguous name, and is not an oracle.

## Frozen request

The model is `models/gemma4-e4b-it-google-qat-gguf/gemma-4-E4B_q4_0-it.gguf`, running on six pinned Intel Core i7-12700 CPUs (`taskset -c 0-5`, `GOMAXPROCS=6`). The request has these exact boundaries:

* Prompt preparation supplies 123 copies of token `10979`; Gemma4 adds BOS token `2`, giving 124 evaluated prompt tokens. `BeginPrefill` allocates and prepares the request outside the timed section, while `PrefillNext(124)` is the prompt measurement.
* Prefill returns a boundary token before generation timing begins. The request exposes 48 output tokens, but only 47 subsequent model evaluations are timed.
* The frozen Go trajectory is `[70990 4941 159371 1390 67109 1485 237064 9120 70846 9120 13063 1390 236743 40594 1390 238007 1390 238122 4829 1390 1390 237307 1390 113858 9222 1390 9222 246293 9222 1390 1390 9222 783 82106 9222 246293 236929 4862 237275 1390 237083 237249 244359 238280 236929 395 236782 236929]`. The real-model gate checks all 48 IDs.

llama.cpp does not select that trajectory from the repeated-token prompt--its first greedy token is `10979`, whereas Go's is `70990`. For a useful performance comparison, the temporary b607 audit driver replays the 47 Go-selected pending tokens after evaluating the same prompt. It scans and synchronises logits on every step but ignores llama.cpp's selected ID. This measures identical matrix shapes and token-dependent MoE routing without pretending that the two implementations have cross-runtime logit parity.

## The CPU-only oracle

The audit build has no GPU backend (`"backends": "CPU"`, `"gpu_info": ""`, `n_gpu_layers: 0`). It uses a context sized for the combined 124-prompt/47-evaluation request, CPU flash-attention auto-selection, and F32 K/V storage. b607's command-line parser does not accept `f32` for `-ctk` or `-ctv`, so the audit source sets `GGML_TYPE_F32` as the local default before rebuilding.

`llama-bench` normally adds its independent prompt and generation cases when `-pg` is supplied. Passing `-p 0 -n 0` leaves only the combined phase:

```bash
taskset -c 0-5 /workspace/tmp/llama-b607-cpu/build-cpu/bin/llama-bench \
  -m "$MODEL" -p 0 -n 0 -pg 124,47 -r 3 -t 6 -ngl 0 -fa auto -o json
```

Six F32-KV samples from two adjacent runs are pooled because this host has conspicuous shared-load outliers. The median is the mean of samples three and four after sorting:

| Phase | Sample times (s, sorted) | Accepted median | 98% gate |
|---|---|---:|---:|
| 124-token prompt | 1.2766, 1.3168, 1.3277, 1.3907, 1.4274, 1.4283 | 91.2296 tok/s | 89.4050 tok/s |
| 47 generation evaluations | 4.3892, 4.3987, 4.4116, 4.5183, 6.7104, 7.5740 | 10.5265 tok/s | 10.3159 tok/s |

The temporary source changes are captured in [`audit/llama-b607-exact-cpu.patch`](audit/llama-b607-exact-cpu.patch). Raw phase telemetry is in [`audit/cpu_exact_go_trajectory_f32kv_124x47_r3.log`](audit/cpu_exact_go_trajectory_f32kv_124x47_r3.log) and [`audit/cpu_exact_go_trajectory_f32kv_paired_r3.log`](audit/cpu_exact_go_trajectory_f32kv_paired_r3.log); the adjacent JSON files record the b607 revision, CPU backend, K/V types and context parameters. The 6.71 s and 7.57 s generation samples are why a single pretty run is not accepted here.

## Where Go now lands

The retained long-prefill path quantises Q8_0 activations across six contiguous worker spans, then feeds the one-weight-row/eight-token SoA AVX-VNNI kernel. Block values remain byte-identical to serial `quantizeQ8_0To`; the focused exact test compares every scale and quantised byte, and running that worker-spawning test under the race detector covers concurrent writes into disjoint spans. That activation work previously established a 38.808 prompt tok/s median.

The retained kernel now leaves the packed Q4 nibbles unsigned, computes `8*sum(Q8)` with a second `VPDPBUSD`, and subtracts that correction from the unsigned dot in each int32 lane before the existing FP32 scale/FMA sequence. This is algebraically identical to `(q4-8)*q8`, but removes the `VPSUBB` Q4 offset and eight per-token `VPABSB`/`VPSIGNB` transforms. The eight FP32 lane accumulators, blockwise FMA and final reduction order do not change. Randomised 1--80-block exact comparisons passed repeatedly.

A tightly paired eight-sample kernel comparison put the new path at 0.831--0.836 of the prior kernel time, a stable approximately 16.7% reduction. In the complete frozen request, three alternating baseline/candidate pairs produced baseline prompt rates of 40.160, 32.624 and 41.281 tok/s and candidate rates of 49.760, 48.875 and 34.350 tok/s. Their medians are **40.160 versus 48.875 tok/s (+21.7%)**. All 48 output IDs remained unchanged. Generation is outside this B64+ projection change and was effectively flat: 9.128 baseline versus 9.161 candidate median eval tok/s. Raw evidence is in [`audit/go_q4_unsigned_correction_microbench.log`](audit/go_q4_unsigned_correction_microbench.log) and [`audit/go_prefill_unsigned_q4_correction_paired_r3.log`](audit/go_prefill_unsigned_q4_correction_paired_r3.log).

| Runtime | Prompt | Generation | Oracle efficiency |
|---|---:|---:|---:|
| CPU-only llama.cpp b607, exact phases | 91.230 tok/s | 10.526 eval tok/s | 100% / 100% |
| Go, retained median | 48.875 tok/s | 9.161 eval tok/s | 53.6% / 87.0% |

The exact workload is therefore not within the 2% target. Decode is 13.0% below the oracle; prefill remains 46.4% below it.

## What the llama.cpp audit changed

b607's AVX2 prompt path packs eight Q4_0 weight rows together and evaluates groups of input rows against them. This raises weight reuse and accumulates each output row with a different vector/reduction topology. Go's exact contract instead requires one YMM accumulator per token, with eight K-lane FP32 partials preserved through every block and reduced only at the end. Under that constraint, the tested llama-style eight-weight-row/one-token orientation was approximately 1.35 times slower than Go's one-weight-row/eight-token orientation. Copying the tile shape is therefore not an exact or faster substitute.

The useful transferable detail was llama.cpp's willingness to dot nibbles without first materialising signed Q4 bytes. On Alder Lake, AVX-VNNI provides unsigned-by-signed `VPDPBUSD`, not AVX-VNNI-INT8's signed-byte `VPDPBSSD`. The dynamic correction above exploits the available instruction while retaining Go's reduction topology.

Persistently storing a llama-style eight-row Q4 packing would not reduce payload size: eight normal Q4_0 blocks occupy 144 bytes and the packed tile also occupies 144 bytes. Keeping it beside `QuantMatrix.Raw` would duplicate all affected Q4_0 weights. This GGUF has 342 Q4_0 tensors occupying 2,219,212,800 bytes (2.066803 GiB), so a complete duplicate would add the same 2.066803 GiB before allocator metadata and alignment. Replacement-only packing would avoid the duplicate but would need pervasive ownership, decode and fallback changes, and the exact-kernel benchmark already rejects its orientation.

## Why the prompt gap is stubborn

The retained path now includes native AVX2/VNNI byte dots, dynamic unsigned-Q4 correction, row grouping for decode, one-row/eight-token activation tiling for prefill, SoA Q8 activation layout, six-way static row scheduling, vectorised Q6 coefficient expansion and parallel contiguous activation quantisation. Four-row/two-token, two-row/four-token, flat-interleaved activation and persistent-worker variants were measured and removed because they regressed the complete request even when an isolated kernel looked promising.

The phase-specific profile that selected this target attributed 83.48% of flat CPU samples to `dotQ4_0Q8_0Tokens8SoAVNNI` before dynamic correction was retained. A new phase-only profile of the retained binary puts the function at 10.87 of 13.28 sampled CPU seconds, or 81.85% flat. `quantizeQ8_0To` is only 4.07% and the four-token tail kernel is 4.74%. The profiled request coincided with severe shared-host load and its 19.156 prompt tok/s is not accepted as throughput; [`audit/go_prefill_retained_unsigned_q4.pprof`](audit/go_prefill_retained_unsigned_q4.pprof) and [`audit/go_prefill_retained_unsigned_q4_profile.log`](audit/go_prefill_retained_unsigned_q4_profile.log) are attribution evidence only.

The source layouts explain that concentration. For this 124-token shape, Go's ordinary Q4 rows and eight-token Q8 tiles require a minimum 38,016 input bytes per QK block to form 992 outputs across eight weight rows--38.32 bytes per output. b607's eight-row Q4 packing, seven sixteen-token AVX2 supertiles and three four-token tails require 5,656 bytes--5.70 bytes per output. This 6.7 times lower payload is not a cache-counter measurement, but it exposes the structural reuse: llama.cpp loads each packed activation for eight weight rows and uses vector lanes as independent outputs. Go reloads activations per weight row because each exact output owns a YMM vector containing its eight persistent FP32 K-lane partials.

AVX2's sixteen-register limit makes that reduction contract consequential. The eight live output vectors plus Q4 unpack, Q8, correction, scale and dot temporaries consume the register file. llama.cpp instead completes each 32-element integer block dot before one FP32 FMA, which permits much wider row/token tiles but differs from Go in 877 of 1,000 deterministic random probes. The dominant retained prefill gap is therefore the exact Q4/Q8 output and reduction orientation--especially repeated activation loads across weight rows--rather than worker scheduling, allocation or generic Go overhead.

Applying the fresh 81.85% share to the accepted 48.875 tok/s median estimates 2.0766 seconds in the tile and 0.4605 seconds elsewhere. If the remainder stays fixed, reaching the 98% prompt gate requires approximately 2.24 times hotspot speed; matching the oracle requires 2.31 times, a 56.7% hotspot-time reduction. Sample share is not wall-clock decomposition, but the estimate rules out another single-digit peripheral win.

Splitting the eight-token tile into two four-token passes freed four YMM registers and allowed two VNNI dependency chains to overlap. It remained exact, but longer pinned samples regressed from an 812 ns median to 1,009 ns because every Q4 block had to be loaded and decoded twice. A compact precomputed `int16` correction looked better in isolation--0.937--0.958 of dynamic-correction kernel time--but enlarged each 80-block activation tile from 23,040 to 33,280 bytes and added packing work. It lost every complete-request pair; medians were 40.392 versus 44.863 tok/s for compact versus dynamic correction, so it was reverted. Evidence is in [`audit/go_q4_compact_correction_microbench.log`](audit/go_q4_compact_correction_microbench.log) and [`audit/go_prefill_compact_correction_rejected_paired_r3.log`](audit/go_prefill_compact_correction_rejected_paired_r3.log).

The lane-transposed four-weight-row/two-token experiment was implemented and rejected. Size-preserving test packers produce one 72-byte four-row Q4 panel and one 72-byte two-token Q8 panel per block. Eight accumulators retain one legacy K lane across the token-major outputs, and a final exact 8-by-8 transpose restores one vector per output before the unchanged reduction. One hundred deterministic random cases covering 1--80 blocks matched every one of the 64 FP32 lane states, all eight outputs and the existing output-major kernel bit-for-bit.

Exactness did not translate into speed. The optimistic assembly-only path performs the final transpose but excludes both Go reduction and packing. Across five pinned samples its single-tile median was 3,574 ns versus 615.5 ns retained; its synthetic 128-row/124-token projection median was 6.285534 ms versus 1.876609 ms retained. That is 0.299 times retained speed, or 3.35 times slower, rather than the required 2.24 times speed-up. The complete experimental wrapper projection median was slower again at 7.805817 ms. Even the optimistic candidate's fastest projection sample was 1.98 times the retained path's slowest, and it would need a further 7.50 times improvement to cross the gate. The blockwise 8-by-8 transpose needed to preserve all legacy lane sequences overwhelms the 52.9% source-payload reduction. Raw results are [`audit/go_q4_lane_transposed_exact.log`](audit/go_q4_lane_transposed_exact.log), [`audit/go_q4_lane_transposed_tile_bench.log`](audit/go_q4_lane_transposed_tile_bench.log) and [`audit/go_q4_lane_transposed_projection_bench.log`](audit/go_q4_lane_transposed_projection_bench.log).

No production projection, model packing or full-request checkpoint was changed, and no second persistent 2.067 GiB representation was created. The one-row/eight-token tile with dynamic correction remains the fastest exact arrangement measured on this AVX2/VNNI machine. Wider reordering is not promoted unless all activations, logits, tokens and K/V state remain exact.

## Reproducing the Go side

```bash
taskset -c 0-5 env GOMAXPROCS=6 \
  go test ./loader/gguf \
  -run 'Test(LaneTransposed|DotQ4_0Q8_0Rows4Tokens2LaneTransposed)' \
  -count=1

taskset -c 0-5 env GOMAXPROCS=6 \
  go test ./loader/gguf -run '^$' \
  -bench '^BenchmarkDotQ4_0Q8_0LaneTransposed(Tiles|Projection)$' \
  -benchmem -benchtime=1s -count=5

MODEL="$PWD/models/gemma4-e4b-it-google-qat-gguf/gemma-4-E4B_q4_0-it.gguf"
taskset -c 0-5 env \
  GOMAXPROCS=6 GOTMPDIR="$PWD/.gotmp" \
  GO_PHERENCE_GEMMA4_GAP_REAL=1 \
  GO_PHERENCE_GEMMA4_MAIN="$MODEL" \
  go test ./model -run '^TestGemma4RealCPUGap124x48$' \
  -count=3 -v -timeout=10m
```

The retained checkpoint passes focused exact Q4/Q8 tests, focused `go vet`, serialised race tests, exact real-model trajectory checks, `git diff --check`, and arm64/riscv64 compile gates. Repository-wide `go test ./...` and `go vet ./...` remain blocked by unrelated pre-existing SpacemiT, diffusion, assembly, unsafe-pointer and Vulkan failures; those failures are tracked separately rather than attributed to this work. CUDA work stays out of this programme until the CPU path reaches the 2% gate--otherwise GPU numbers merely make the unresolved CPU kernel harder to see.
